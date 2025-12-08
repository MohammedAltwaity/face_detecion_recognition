#!/usr/bin/env python3
"""
Raspberry Pi Image Sender
=========================

Monitors 'extracted_faces' folder for new images and sends ALL of them to the server.
Only sends images created after the script starts running.
Ensures no image is left without being sent.
"""

import os
import sys
import json
import time
import logging
import requests
from datetime import datetime
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rpi_image_sender.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RPiImageSender:
    def __init__(self):
        # Watch extracted_faces folder (created by update5_with_time_delay.py)
        self.watch_folder = Path('extracted_faces')
        self.config_file = Path('config.json')
        self.script_start_time = datetime.now()
        self.supported_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
        self.processed_files = set()
        self.pending_files = set()  # Track files that need to be sent
        
        # Load configuration
        self.config = self.load_config()
        
        # Ensure watch folder exists
        self.watch_folder.mkdir(exist_ok=True)
        
        logger.info(f"Script started at: {self.script_start_time}")
        logger.info(f"Watching folder: {self.watch_folder.absolute()}")
        logger.info(f"Server URL: {self.config['server_url']}")
        logger.info(f"Send delay between images: {self.config.get('send_delay', 0.5)} seconds")
        
        # Process any existing images created after script start
        self.scan_and_queue_existing_images()
    
    def load_config(self):
        """Load configuration from JSON file"""
        default_config = {
            "server_url": "http://10.109.225.88:5000/api/upload-image",
            "server_timeout": 30,
            "supported_formats": [".jpg", ".jpeg", ".png", ".gif", ".bmp"],
            "check_interval": 2,
            "send_delay": 0.5  # Delay in seconds between sending each image (default: 0.5s)
        }
        
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    logger.info("Configuration loaded successfully")
                    return config
            else:
                logger.info("No config file found, using defaults")
                # Save default config
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=2)
                return default_config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return default_config
    
    def is_supported_image(self, file_path):
        """Check if file is a supported image format"""
        return file_path.suffix.lower() in self.supported_extensions
    
    def is_file_ready(self, file_path):
        """Check if file is fully written and ready for processing"""
        try:
            if not file_path.exists() or not file_path.is_file():
                return False
            
            if file_path.stat().st_size == 0:
                return False
            
            # Check if file is still being written
            try:
                with open(file_path, 'rb') as f:
                    f.read(1)
                return True
            except (PermissionError, OSError):
                return False
        except Exception:
            return False
    
    def wait_for_file_ready(self, file_path, max_wait=30):
        """Wait for file to be fully written"""
        logger.info(f"Waiting for file to be ready: {file_path.name}")
        
        start_time = time.time()
        while time.time() - start_time < max_wait:
            if self.is_file_ready(file_path):
                logger.info(f"File ready: {file_path.name}")
                return True
            time.sleep(1)
        
        logger.warning(f"File not ready after {max_wait}s: {file_path.name}")
        return False
    
    def get_all_new_images(self):
        """Get ALL images created after script start that haven't been processed"""
        try:
            new_images = []
            
            for file_path in self.watch_folder.iterdir():
                if (file_path.is_file() and 
                    self.is_supported_image(file_path) and
                    str(file_path) not in self.processed_files):
                    
                    # Check creation time
                    creation_time = datetime.fromtimestamp(file_path.stat().st_ctime)
                    if creation_time > self.script_start_time:
                        new_images.append(file_path)
            
            # Sort by creation time (oldest first to process in order)
            new_images.sort(key=lambda x: x.stat().st_ctime)
            return new_images
            
        except Exception as e:
            logger.error(f"Error finding new images: {e}")
            return []
    
    def scan_and_queue_existing_images(self):
        """Scan folder for existing images created after script start and queue them"""
        logger.info("Scanning for existing images created after script start...")
        new_images = self.get_all_new_images()
        
        if new_images:
            logger.info(f"Found {len(new_images)} new image(s) to process")
            for img_path in new_images:
                self.pending_files.add(str(img_path))
                logger.info(f"Queued: {img_path.name}")
        else:
            logger.info("No new images found")
    
    def send_image_to_server(self, image_path):
        """Send image to the server via HTTP POST"""
        try:
            logger.info(f"Sending image to server: {image_path.name}")
            
            # Prepare the request
            url = self.config['server_url']
            timeout = self.config.get('server_timeout', 30)
            
            with open(image_path, 'rb') as f:
                files = {'image': f}
                response = requests.post(url, files=files, timeout=timeout)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('status') == 'success':
                    logger.info(f"Image sent successfully: {result['filename']}")
                    logger.info(f"Server confirmation: {result['received_at']}")
                    return True
                else:
                    logger.error(f"Server returned error: {result.get('message', 'Unknown error')}")
                    return False
            else:
                logger.error(f"Server error: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.ConnectionError:
            logger.error("Connection failed - server unreachable")
            return False
        except requests.exceptions.Timeout:
            logger.error("Request timed out")
            return False
        except Exception as e:
            logger.error(f"Error sending image: {e}")
            return False
    
    def process_all_pending_images(self):
        """Process ALL pending images, ensuring none are left behind"""
        # Get all new images that haven't been processed
        new_images = self.get_all_new_images()
        
        # Add to pending set
        for img_path in new_images:
            self.pending_files.add(str(img_path))
        
        if not self.pending_files:
            return 0
        
        processed_count = 0
        failed_files = []
        send_delay = self.config.get('send_delay', 0.5)  # Get delay from config
        
        # Convert to list and sort by creation time to process in order
        pending_list = sorted(
            [Path(f) for f in self.pending_files if Path(f).exists()],
            key=lambda x: x.stat().st_ctime
        )
        
        # Process all pending files
        for idx, file_path in enumerate(pending_list):
            file_path_str = str(file_path)
            
            if not file_path.exists():
                logger.warning(f"File no longer exists: {file_path.name}")
                self.pending_files.discard(file_path_str)
                continue
            
            # Wait for file to be ready
            if not self.wait_for_file_ready(file_path):
                logger.error(f"File not ready: {file_path.name}")
                failed_files.append(file_path_str)
                continue
            
            # Send to server
            success = self.send_image_to_server(file_path)
            
            if success:
                # Mark as processed
                self.processed_files.add(file_path_str)
                self.pending_files.discard(file_path_str)
                processed_count += 1
                logger.info(f"✓ Image processed successfully: {file_path.name} ({processed_count}/{len(pending_list)})")
                
                # Add delay between sending images (except after the last one)
                if idx < len(pending_list) - 1 and send_delay > 0:
                    logger.debug(f"Waiting {send_delay}s before sending next image...")
                    time.sleep(send_delay)
            else:
                logger.error(f"✗ Failed to process image: {file_path.name}")
                failed_files.append(file_path_str)
                # Still add delay even on failure to avoid rapid retries
                if idx < len(pending_list) - 1 and send_delay > 0:
                    time.sleep(send_delay)
        
        # Keep failed files in pending for retry
        if failed_files:
            logger.warning(f"{len(failed_files)} image(s) failed, will retry later")
        
        return processed_count
    
    def process_latest_image(self):
        """Process the latest image if available (for backward compatibility)"""
        return self.process_all_pending_images() > 0

class ImageFolderWatcher(FileSystemEventHandler):
    def __init__(self, sender):
        self.sender = sender
    
    def on_created(self, event):
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        if not self.sender.is_supported_image(file_path):
            return
        
        logger.info(f"New image detected: {file_path.name}")
        
        # Add to pending and process all pending images
        self.sender.pending_files.add(str(file_path))
        self.sender.process_all_pending_images()
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        if (self.sender.is_supported_image(file_path) and 
            str(file_path) not in self.sender.processed_files):
            
            if self.sender.is_file_ready(file_path):
                logger.info(f"File ready after modification: {file_path.name}")
                # Add to pending and process all pending images
                self.sender.pending_files.add(str(file_path))
                self.sender.process_all_pending_images()

def main():
    logger.info("=" * 60)
    logger.info("RASPBERRY PI IMAGE SENDER")
    logger.info("=" * 60)
    
    # Initialize sender
    sender = RPiImageSender()
    
    # Setup folder watcher
    event_handler = ImageFolderWatcher(sender)
    observer = Observer()
    observer.schedule(event_handler, str(sender.watch_folder), recursive=False)
    
    try:
        # Start watching
        observer.start()
        logger.info(f"Started monitoring: {sender.watch_folder.absolute()}")
        logger.info("Press Ctrl+C to stop")
        
        # Main loop - periodically process all pending images
        while True:
            time.sleep(sender.config.get('check_interval', 2))
            # Process all pending images to ensure none are left behind
            processed = sender.process_all_pending_images()
            if processed > 0:
                logger.info(f"Processed {processed} image(s) in this cycle")
            
    except KeyboardInterrupt:
        logger.info("Stopping image sender...")
        observer.stop()
    
    observer.join()
    logger.info("Image sender stopped")

if __name__ == "__main__":
    main()
