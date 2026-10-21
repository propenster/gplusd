import time
import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

def transform(df):
    df = df.rename(columns={
        "Regulated Gene": "gene_id",
        "Absolute position": "coordinates_raw",
        "Binding seq.(cis-element)": "sequence"
    })
    data = []
    pos_coords = []
    
    for _, row in df.iterrows():
        coord_str = str(row["coordinates_raw"]).replace("..", "-")
        if pd.isna(row["sequence"]) or row["sequence"] in ["", "ND", "None"]:
            continue
            
        # Ensure we can split the coordinate string properly
        coords = coord_str.split("-")
        if len(coords) < 2:
            continue
            
        try:
            start, end = int(coords[0]), int(coords[-1])
            pos_coords.append((start, end))
            
            data.append({
                "gene_id": row["gene_id"],
                "gene_name": row["gene_id"],
                "coordinates": coord_str,
                "sequence": row["sequence"],
                "label": 1
            })
        except ValueError:
            # Skip rows where coordinates are malformed text instead of numbers
            continue
            
    return pd.DataFrame(data), pos_coords

def collect_dbtbs_data():
    sigmas = ["SigA", "SigB", "SigD", "SigE", "SigF", "SigG", "SigH", "SigK", "SigM", "SigW", "SigX", "SigY"]
    base_url = "https://dbtbs.hgc.jp/COG/tfac"
    all_sigma_data = []
    all_positive_coords = []
    good_cols = ["Regulated Gene", "Absolute position", "Binding seq.(cis-element)"]
    
    for sigma in sigmas:
        print(f"Running collect data for sigma {sigma}")
        try:
            df = pd.read_html(f"{base_url}/{sigma}.html")
            data_table = max(df, key=len)
            
            if not all(col in data_table.columns for col in good_cols):
                print(f"  Skipping {sigma}: missing required columns.")
                continue
                
            data_table = data_table[good_cols].copy()
            clean_df, pos_coords = transform(data_table)
            
            # Track which sigma factor this is for FASTA headers
            clean_df["sigma"] = sigma 
            
            all_sigma_data.append(clean_df)
            all_positive_coords.extend(pos_coords)
            
        except Exception as e:
            print(f"  Error while loading data for {sigma}: {e}")
        
        time.sleep(1)
    
    master_df = pd.concat(all_sigma_data, ignore_index=True)
    print(f"\nSuccessfully collected {len(master_df)} valid operons/sequences.")
    return master_df, all_positive_coords

def extract_dbtbs_promoters(promoter_df, pos_coords, genome_fasta, output_fasta):
    # Load the B. subtilis reference genome
    genome_record = next(SeqIO.parse(genome_fasta, "fasta"))
    genome_seq = genome_record.seq
    extracted_records = []
    
    for idx, row in promoter_df.iterrows():
        if idx >= len(pos_coords):
            break
            
        gene_name = str(row['gene_id']).replace(" ", "_").replace("/", "-")
        sigma = row['sigma']
        
        # Use the extracted coordinates from the transform function
        start_pos, end_pos = pos_coords[idx]
        
        # Convert 1-based biological coordinate to 0-based Python index
        anchor_index = start_pos - 1
        
        try:
            if start_pos < end_pos:
                # Plus strand: -60 upstream, +20 downstream
                slice_start = anchor_index - 60
                slice_end = anchor_index + 21
                segment = genome_seq[slice_start:slice_end]
            else:
                # Minus strand (start > end): Reverse the upstream/downstream logic
                slice_start = anchor_index - 20
                slice_end = anchor_index + 61
                segment = genome_seq[slice_start:slice_end].reverse_complement()
                
            if len(segment) == 81 and "N" not in segment.upper():
                record = SeqRecord(
                    segment,
                    id=f"{gene_name}_{sigma}_{start_pos}",
                    description="DBTBS_benchmark_81bp"
                )
                extracted_records.append(record)
                
        except IndexError:
            continue
            
    SeqIO.write(extracted_records, output_fasta, "fasta")
    print(f"Successfully extracted {len(extracted_records)} formatted sequences to {output_fasta}")


# def extract_correct_dbtbs_promoters(promoter_df, genome_fasta, output_fasta):
#     genome_record = next(SeqIO.parse(genome_fasta, "fasta"))
#     genome_seq = genome_record.seq
#     extracted_records = []
    
#     for idx, row in promoter_df.iterrows():
#         gene_name = str(row['gene_id']).strip().replace(" ", "_").replace("/", "-")
#         sigma = row['sigma']
#         binding_seq = str(row['sequence']).upper()
        
#         # 1. Parse coordinates (FIXED: use 'coordinates' instead of 'coordinates_raw')
#         coords_raw = str(row['coordinates']).replace("..", "-").split("-")
#         try:
#             pos1, pos2 = int(coords_raw[0]), int(coords_raw[-1])
#         except ValueError:
#             continue
            
#         # 2. Extract the raw genomic locus (0-based)
#         start_idx = pos1 - 1
#         end_idx = pos2
#         locus_fwd = genome_seq[start_idx:end_idx].upper()
        
#         # 3. Deduce strand by aligning the DBTBS binding sequence to the genome
#         # If the known binding sequence matches the reverse complement of the locus, it's on the minus strand.
#         strand = '+'
#         if binding_seq in locus_fwd.reverse_complement():
#             strand = '-'
            
#         # 4. Handle strand orientation & slicing correctly
#         if strand == '+':
#             tss_estimate = pos1  
#             slice_start = (tss_estimate - 1) - 60
#             slice_end = (tss_estimate - 1) + 21
#             segment = genome_seq[slice_start:slice_end]
#         else:
#             tss_estimate = pos2
#             slice_start = (tss_estimate - 1) - 20
#             slice_end = (tss_estimate - 1) + 61
#             segment = genome_seq[slice_start:slice_end].reverse_complement()
            
#         if len(segment) == 81 and "N" not in segment.upper():
#             rec = SeqRecord(
#                 segment,
#                 id=f"{gene_name}_{sigma}_{tss_estimate}",
#                 description="DBTBS_benchmark_81bp"
#             )
#             extracted_records.append(rec)
            
#     SeqIO.write(extracted_records, output_fasta, "fasta")
#     print(f"Successfully extracted {len(extracted_records)} correctly oriented sequences to {output_fasta}")


def extract_correct_dbtbs_promoters(promoter_df, genome_fasta, output_fasta):
    genome_record = next(SeqIO.parse(genome_fasta, "fasta"))
    genome_seq = genome_record.seq
    extracted_records = []
    
    for idx, row in promoter_df.iterrows():
        gene_name = str(row['gene_id']).strip().replace(" ", "_").replace("/", "-")
        sigma = row['sigma']
        binding_seq = str(row['sequence']).upper()
        
        # 1. Parse coordinates
        coords_raw = str(row['coordinates']).replace("..", "-").split("-")
        try:
            pos1, pos2 = int(coords_raw[0]), int(coords_raw[-1])
        except ValueError:
            continue
            
        # 2. Extract the raw genomic locus safely (handle max..min backwards notation)
        p_min, p_max = min(pos1, pos2), max(pos1, pos2)
        locus_fwd = genome_seq[p_min-1 : p_max].upper()
        
        # 3. Deduce strand
        strand = '+'
        if binding_seq in locus_fwd.reverse_complement():
            strand = '-'
            
        # 4. Estimate true TSS position to center the binding site at index ~25 (-35 box)
        if strand == '+':
            # 5' end of binding site is p_min. Estimated TSS is ~35bp downstream.
            tss_estimate = p_min + 35  
            slice_start = (tss_estimate - 1) - 60
            slice_end = (tss_estimate - 1) + 21
            segment = genome_seq[slice_start:slice_end]
        else:
            # 5' end of binding site on minus strand is p_max. Estimated TSS is ~35bp downstream (minus direction).
            tss_estimate = p_max - 35
            slice_start = (tss_estimate - 1) - 20
            slice_end = (tss_estimate - 1) + 61
            segment = genome_seq[slice_start:slice_end].reverse_complement()
            
        if len(segment) == 81 and "N" not in segment.upper():
            rec = SeqRecord(
                segment,
                id=f"{gene_name}_{sigma}_{tss_estimate}",
                description="DBTBS_benchmark_81bp"
            )
            extracted_records.append(rec)
            
    SeqIO.write(extracted_records, output_fasta, "fasta")
    print(f"Successfully extracted {len(extracted_records)} correctly aligned sequences to {output_fasta}")

# Execute the pipeline
df, _ = collect_dbtbs_data()
# We drop the 'coords' argument entirely and let the function deduce the strand automatically
extract_correct_dbtbs_promoters(df, "bsubtilis_genome.fasta", "dbtbs_benchmark.fasta")
