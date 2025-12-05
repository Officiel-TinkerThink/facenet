import os
import gdown
import zipfile
import requests
from tqdm import tqdm

class CelebADownloader:
    """
    Downloads and prepares CelebA dataset from Google Drive
    """
    
    # Google Drive file IDs (public links)
    DRIVE_IDS = {
        'img_align_celeba': '0B7EVK8r0v71pZjFTYXZWM3FlRnM',  # Main images
        'list_eval_partition': '0B7EVK8r0v71pY0NSMzRuSXJEVkk',  # Train/val/test split
        'list_bbox_celeba': '0B7EVK8r0v71peklHb0pGdDl6R28',  # Bounding boxes
        'list_landmarks_align_celeba': '0B7EVK8r0v71pTzJIdlJWdHczRlU',  # Landmarks
        'list_attr_celeba': '0B7EVK8r0v71pblRyaVFSWGxPY0U',  # Attributes
        'identity_CelebA': '1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS',  # Identity labels
    }
    
    # File sizes for progress tracking (approx)
    FILE_SIZES = {
        'img_align_celeba': 1.34 * 1024**3,  # 1.34 GB
        'list_eval_partition': 100 * 1024,  # 100 KB
        'list_bbox_celeba': 200 * 1024,  # 200 KB
        'list_landmarks_align_celeba': 200 * 1024,  # 200 KB
        'list_attr_celeba': 2 * 1024**2,  # 2 MB
        'identity_CelebA': 400 * 1024,  # 400 KB
    }
    
    def __init__(self, output_dir='data/celeba'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def download_file(self, file_key, force=False):
        """Download a single file from Google Drive"""
        file_id = self.DRIVE_IDS[file_key]
        filename = file_key + ('.zip' if file_key == 'img_align_celeba' else '.txt')
        filepath = os.path.join(self.output_dir, filename)
        
        # Skip if already exists
        if os.path.exists(filepath) and not force:
            print(f"✓ {filename} already exists, skipping...")
            return filepath
        
        # Google Drive download URL
        url = f'https://drive.google.com/uc?id={file_id}&confirm=t'
        
        print(f"Downloading {filename}...")
        
        # Download with progress bar
        try:
            gdown.download(url, filepath, quiet=False)
            print(f"✓ Downloaded {filename}")
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")
            
            # Fallback: Try direct download with requests
            print("Trying fallback download method...")
            self._download_with_requests(url, filepath)
        
        return filepath
    
    def _download_with_requests(self, url, filepath):
        """Fallback download method"""
        session = requests.Session()
        
        try:
            response = session.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            
            with open(filepath, 'wb') as f, tqdm(
                desc=os.path.basename(filepath),
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(block_size):
                    f.write(data)
                    pbar.update(len(data))
            
            print(f"✓ Downloaded via fallback: {os.path.basename(filepath)}")
            
        except Exception as e:
            print(f"✗ Fallback also failed: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)
            raise
    
    def extract_images(self, zip_path):
        """Extract the main image archive"""
        img_dir = os.path.join(self.output_dir, 'img_align_celeba')
        
        if os.path.exists(img_dir) and len(os.listdir(img_dir)) > 0:
            print("✓ Images already extracted, skipping...")
            return img_dir
        
        print(f"Extracting images from {os.path.basename(zip_path)}...")
        
        # Create directory
        os.makedirs(img_dir, exist_ok=True)
        
        # Extract with progress bar
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get total files for progress
            file_list = zip_ref.infolist()
            
            # Extract each file
            for file in tqdm(file_list, desc="Extracting"):
                zip_ref.extract(file, img_dir)
        
        print(f"✓ Extracted {len(os.listdir(img_dir))} images")
        return img_dir
    
    def create_identity_file(self):
        """
        Create identity_CelebA.txt if it doesn't exist
        Alternative: Download or generate
        """
        identity_path = os.path.join(self.output_dir, 'identity_CelebA.txt')
        
        if os.path.exists(identity_path):
            print("✓ identity_CelebA.txt already exists")
            return identity_path
        
        print("Creating identity_CelebA.txt...")
        
        # Method 1: Try to download
        try:
            identity_path = self.download_file('identity_CelebA', force=False)
            print("✓ Downloaded identity file")
            return identity_path
        except:
            print("Could not download identity file, creating synthetic one...")
        
        # Method 2: Create synthetic identity mapping
        img_dir = os.path.join(self.output_dir, 'img_align_celeba')
        if not os.path.exists(img_dir):
            raise ValueError("Image directory not found. Extract images first.")
        
        # Get all image files
        image_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        
        # Create identity mapping: group images into "pseudo-people"
        # Each pseudo-person gets 20 images
        images_per_person = 20
        with open(identity_path, 'w') as f:
            person_id = 0
            count = 0
            
            for img_name in image_files[:4000]:  # First 4000 images for learning
                f.write(f"{img_name} {person_id}\n")
                count += 1
                
                if count >= images_per_person:
                    person_id += 1
                    count = 0
        
        print(f"✓ Created synthetic identity file with {person_id} people")
        return identity_path
    
    def verify_dataset(self):
        """Verify all required files exist"""
        required_files = [
            ('img_align_celeba', 'directory'),
            ('list_eval_partition.txt', 'file'),
            ('list_attr_celeba.txt', 'file'),
            ('identity_CelebA.txt', 'file'),
        ]
        
        print("\n" + "="*50)
        print("Dataset Verification")
        print("="*50)
        
        all_ok = True
        for name, type_ in required_files:
            path = os.path.join(self.output_dir, name)
            
            if type_ == 'directory':
                exists = os.path.isdir(path) and len(os.listdir(path)) > 0
                status = "✓" if exists else "✗"
                print(f"{status} {name}: {'Exists with files' if exists else 'Missing or empty'}")
                
            else:  # file
                exists = os.path.isfile(path) and os.path.getsize(path) > 0
                status = "✓" if exists else "✗"
                size = f" ({os.path.getsize(path)/1024:.1f} KB)" if exists else ""
                print(f"{status} {name}: {'Exists' + size if exists else 'Missing'}")
            
            if not exists:
                all_ok = False
        
        print("="*50)
        if all_ok:
            print("✓ Dataset ready for training!")
        else:
            print("✗ Some files are missing")
        
        return all_ok
    
    def download_all(self, skip_images=False):
        """
        Download all CelebA files
        skip_images: Skip the large image download (1.34 GB)
        """
        print("="*60)
        print("CelebA Dataset Downloader")
        print("="*60)
        
        # Download metadata files first
        print("\nDownloading metadata files...")
        metadata_files = [
            'list_eval_partition',
            'list_bbox_celeba',
            'list_landmarks_align_celeba',
            'list_attr_celeba',
        ]
        
        for file_key in metadata_files:
            self.download_file(file_key)
        
        # Download images (optional)
        if not skip_images:
            print("\nDownloading images (this may take a while)...")
            zip_path = self.download_file('img_align_celeba')
            
            print("\nExtracting images...")
            self.extract_images(zip_path)
            
            # Clean up zip file to save space
            if os.path.exists(zip_path):
                os.remove(zip_path)
                print("✓ Removed zip file to save space")
        
        # Create/download identity file
        print("\nSetting up identity labels...")
        self.create_identity_file()
        
        # Verify
        print("\nVerifying dataset...")
        self.verify_dataset()
        
        print("\n" + "="*60)
        print("Download Complete!")
        print(f"Dataset location: {os.path.abspath(self.output_dir)}")
        print("="*60)

# ==================== QUICK DOWNLOAD FUNCTION ====================
def download_celeba_quick(output_dir='CelebA', skip_images=False):
    """
    One-line function to download CelebA
    """
    downloader = CelebADownloader(output_dir)
    downloader.download_all(skip_images=skip_images)
    return downloader.output_dir

# ==================== MINIMAL VERSION ====================
def minimal_download():
    """Minimal version for FaceNet training"""
    import gdown
    import os
    
    output_dir = "CelebA"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Downloading minimal CelebA for FaceNet...")
    
    # 1. Download images (1.34 GB)
    print("\n1. Downloading images...")
    img_url = "https://drive.google.com/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM&confirm=t"
    img_zip = os.path.join(output_dir, "img_align_celeba.zip")
    gdown.download(img_url, img_zip, quiet=False)
    
    # 2. Extract images
    print("\n2. Extracting images...")
    import zipfile
    with zipfile.ZipFile(img_zip, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    os.remove(img_zip)  # Clean up
    
    # 3. Download train/val/test split
    print("\n3. Downloading dataset split...")
    split_url = "https://drive.google.com/uc?id=0B7EVK8r0v71pY0NSMzRuSXJEVkk&confirm=t"
    split_file = os.path.join(output_dir, "list_eval_partition.txt")
    gdown.download(split_url, split_file, quiet=False)
    
    # 4. Create identity file
    print("\n4. Creating identity labels...")
    img_dir = os.path.join(output_dir, "img_align_celeba")
    images = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    
    with open(os.path.join(output_dir, "identity_CelebA.txt"), 'w') as f:
        for i, img_name in enumerate(images[:4000]):  # First 4000 images
            person_id = i // 20  # 20 images per person
            f.write(f"{img_name} {person_id}\n")
    
    print(f"\n✓ Dataset ready at: {output_dir}")
    print(f"   Images: {len(images)}")
    print(f"   People: {200} (synthetic)")

# ==================== COMMAND LINE INTERFACE ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download CelebA dataset')
    parser.add_argument('--output_dir', default='data/celeba', help='Output directory')
    parser.add_argument('--skip_images', action='store_true', 
                       help='Skip downloading images (1.34 GB)')
    parser.add_argument('--quick', action='store_true',
                       help='Use minimal download for FaceNet training')
    
    args = parser.parse_args()
    
    if args.quick:
        minimal_download()
    else:
        downloader = CelebADownloader(args.output_dir)
        downloader.download_all(skip_images=args.skip_images)