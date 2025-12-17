"""
Data Organization Script for HAM10000 Dataset
-----------------------------------------------
This script organizes skin lesion images into disease-specific folders
based on the HAM10000_metadata.csv file.

Author: Dermo-Scope Team
"""

import os
import shutil
import pandas as pd
import zipfile
from pathlib import Path

# Define paths
# PROJECT_ROOT = Path(__file__).parent.parent
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "raw_data"
ORGANIZED_DATA_DIR = PROJECT_ROOT / "organized_data"
METADATA_FILE = RAW_DATA_DIR / "HAM10000_metadata.csv"

# Disease class mapping
DISEASE_CLASSES = {
    'akiec': 'Actinic Keratoses and Intraepithelial Carcinoma',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevi',
    'vasc': 'Vascular Lesions'
}


def unzip_archive():
    """Unzip the archive file to raw_data directory."""
    archive_path = PROJECT_ROOT / "archive.zip"
    
    print("=" * 60)
    print("DERMO-SCOPE: Data Extraction Script")
    print("=" * 60)
    print(f"\n[Step 0/4] Extracting archive...\\n")
    
    # Check if archive exists
    if not archive_path.exists():
        print(f"[X] Archive file not found: {archive_path}")
        print("   Please ensure archive.zip is in the project root folder.")
        return False
    
    # Check if raw_data already has files
    if RAW_DATA_DIR.exists() and any(RAW_DATA_DIR.iterdir()):
        print(f"[!] Raw data directory already contains files: {RAW_DATA_DIR}")
        user_input = input("  Do you want to skip extraction? (y/n): ").strip().lower()
        if user_input == 'y':
            print("  [OK] Skipping extraction, using existing files...")
            return True
    
    # Create raw_data directory if it doesn't exist
    RAW_DATA_DIR.mkdir(exist_ok=True)
    
    # Extract the archive
    try:
        print(f"  Extracting: {archive_path.name}")
        print(f"  Destination: {RAW_DATA_DIR}")
        
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            # Get list of files
            file_list = zip_ref.namelist()
            total_files = len(file_list)
            
            print(f"  Total files in archive: {total_files}")
            print("  Extracting files...")
            
            # Extract all files
            zip_ref.extractall(RAW_DATA_DIR)
            
            print(f"  [OK] Successfully extracted {total_files} files to {RAW_DATA_DIR}")
            return True
            
    except zipfile.BadZipFile:
        print(f"[X] ERROR: {archive_path} is not a valid zip file or is corrupted.")
        return False
    except Exception as e:
        print(f"[X] ERROR: Failed to extract archive: {e}")
        return False


def create_directories():
    """Create organized data directory structure."""
    print("=" * 60)
    print("DERMO-SCOPE: Data Organization Script")
    print("=" * 60)
    print("\n[Step 1/4] Creating directory structure...")
    
    # Create main organized data directory
    ORGANIZED_DATA_DIR.mkdir(exist_ok=True)
    
    # Create subdirectories for each disease class
    for disease_code in DISEASE_CLASSES.keys():
        disease_dir = ORGANIZED_DATA_DIR / disease_code
        disease_dir.mkdir(exist_ok=True)
        print(f"  [OK] Created directory: {disease_code}/")
    
    print(f"\nOrganized data directory: {ORGANIZED_DATA_DIR}")


def load_metadata():
    """Load and validate metadata file."""
    print("\n[Step 2/4] Loading metadata...")
    
    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_FILE}\n"
            f"Please ensure HAM10000_metadata.csv is in the raw_data/ folder."
        )
    
    df = pd.read_csv(METADATA_FILE)
    print(f"  [OK] Loaded {len(df)} records from metadata file")
    
    # Validate required columns
    if 'image_id' not in df.columns or 'dx' not in df.columns:
        raise ValueError(
            "Metadata file must contain 'image_id' and 'dx' columns.\n"
            f"Found columns: {list(df.columns)}"
        )
    
    return df


def organize_images(metadata_df):
    """Organize images into disease-specific folders."""
    print("\n[Step 3/4] Organizing images by disease class...")
    
    total_images = len(metadata_df)
    successful_copies = 0
    missing_images = []
    
    # Statistics per class
    class_counts = {disease: 0 for disease in DISEASE_CLASSES.keys()}
    
    for idx, row in metadata_df.iterrows():
        image_id = row['image_id']
        disease_class = row['dx']
        
        # Try different image extensions
        possible_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        image_found = False
        
        for ext in possible_extensions:
            # Check in multiple possible locations
            possible_paths = [
                RAW_DATA_DIR / f"{image_id}{ext}",
                RAW_DATA_DIR / "HAM10000_images_part_1" / f"{image_id}{ext}",
                RAW_DATA_DIR / "HAM10000_images_part_2" / f"{image_id}{ext}"
            ]
            
            for source_path in possible_paths:
                if source_path.exists():
                    # Copy to organized directory
                    dest_dir = ORGANIZED_DATA_DIR / disease_class
                    dest_path = dest_dir / f"{image_id}{ext}"
                    
                    try:
                        shutil.copy2(source_path, dest_path)
                        successful_copies += 1
                        class_counts[disease_class] += 1
                        image_found = True
                        
                        # Progress indicator
                        if (idx + 1) % 500 == 0:
                            print(f"  Progress: {idx + 1}/{total_images} images processed...")
                        
                        break  # Found the image, break inner loop
                        
                    except Exception as e:
                        print(f"  [!] Error copying {image_id}: {e}")
                        missing_images.append(image_id)
                        break # Error copying, but found it. Stop looking.
            
            if image_found:
                break # Found the image with this extension, stop looking for other extensions
        
        if not image_found:
            missing_images.append(image_id)
    
    return successful_copies, missing_images, class_counts


def print_summary(total, successful, missing_images, class_counts):
    """Print organization summary."""
    print("\n" + "=" * 60)
    print("[Step 4/4] Organization Summary")
    print("=" * 60)
    
    print(f"\n[Stats] Overall Statistics:")
    print(f"  Total images in metadata: {total}")
    print(f"  Successfully organized:   {successful}")
    print(f"  Missing images:           {len(missing_images)}")
    print(f"  Success rate:             {(successful/total)*100:.1f}%")
    
    print(f"\n[Folder] Images per Disease Class:")
    for disease_code, count in sorted(class_counts.items()):
        disease_name = DISEASE_CLASSES[disease_code]
        percentage = (count / successful * 100) if successful > 0 else 0
        print(f"  {disease_code:6s} - {count:5d} images ({percentage:5.1f}%) - {disease_name}")
    
    # Show sample of missing images if any
    if missing_images:
        print(f"\n[!] Missing Images (showing first 10):")
        for img_id in missing_images[:10]:
            print(f"  - {img_id}")
        if len(missing_images) > 10:
            print(f"  ... and {len(missing_images) - 10} more")
    
    # Validation
    if successful < (total * 0.7):
        print("\n[X] WARNING: Less than 70% of images were found!")
        print("   Please check your raw_data/ folder contains the image files.")
    else:
        print("\n[OK] Data organization completed successfully!")


def main():
    """Main execution function."""
    try:
        # Step 0: Unzip archive
        if not unzip_archive():
            print("\n[X] Failed to extract archive. Please check the error messages above.")
            return 1
        
        # Step 1: Create directories
        create_directories()
        
        # Step 2: Load metadata
        metadata_df = load_metadata()
        
        # Step 3: Organize images
        successful, missing_images, class_counts = organize_images(metadata_df)
        
        # Step 4: Print summary
        print_summary(len(metadata_df), successful, missing_images, class_counts)
        
        print("\n" + "=" * 60)
        print("Next Step: Run model training script")
        print("Command: python model_training/02_train_model.py")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[X] ERROR: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure archive.zip is extracted to raw_data/ folder")
        print("2. Verify HAM10000_metadata.csv exists in raw_data/")
        print("3. Check that image files are in raw_data/")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
