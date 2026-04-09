from pathlib import Path
import pandas as pd

def extract_sequences(species_list, folder_name, output_file):
    # Ensure the output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    search_path = Path(f"data/{folder_name}")
    
    if not search_path.exists():
        print(f"Directory not found: {search_path}")
        return

    with open(output_file, "w") as f:
        for spp in species_list:            
            files_found = list(search_path.glob(f"*{spp}*.csv"))
            for file in files_found:
                df = pd.read_csv(file)
                sequences = df["PromoterSeq"].dropna().astype(str)
                for pro in sequences:
                    f.write(f"{pro.upper()}\n")
if __name__ == "__main__":
    gplus = [
        "Bacillus subtilis subsp. subtilis str. 168",
        "Staphylococcus aureus MW2", 
        "Staphylococcus epidermidis ATCC 12228",
        "Streptococcus pyogenes strain S119", 
        "Corynebacterium diphtheriae NCTC 13129", 
        "Corynebacterium glutamicum ATCC 13032", 
        "Paenibacillus riograndensis SBR5"
    ]

    gminus = [
        "Acinetobacter baumannii ATCC 17978",
        "Agrobacterium tumefaciens str C58",
        "Bradyrhizobium japonicum USDA 110", 
        "Burkholderia cenocepacia J2315", 
        "Campylobacter jejuni 81-176", 
        "Campylobacter jejuni 81116", 
        "Campylobacter jejuni NCTC11168", 
        "Campylobacter jejuni RM1221", 
        "Escherichia coli str K-12 substr. MG1655",
        "Helicobacter pylori strain 26695", 
        "Klebsiella aerogenes KCTC 2190", 
        "Nostoc sp. PCC7120", 
        "Onion yellows phytoplasma OY-M", 
        "Pseudomonas putida strain KT2440", 
        "Shigella flexneri 5a str. M90T", 
        "Sinorhizobium meliloti 1021", 
        "Synechococcus elongatus PCC 7942", 
        "Synechocystis sp. PCC 6803", 
        "Xanthomonas campestris pv. campestris B100"
    ]
    
    print("Processing Gram-positive bacteria...")
    extract_sequences(gplus, "gplus", "./data/gplus.txt")
    
    print("\nProcessing Gram-negative bacteria...")
    extract_sequences(gminus, "gminus", "./data/gminus.txt")
    
