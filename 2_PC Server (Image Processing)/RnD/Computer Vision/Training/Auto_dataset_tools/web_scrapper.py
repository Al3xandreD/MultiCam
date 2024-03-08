import os
from google_images_download import google_images_download

def download_images(query, limit=10, output_directory='images'):
    response = google_images_download.googleimagesdownload()

    # Set the parameters for image download
    arguments = {
        "keywords": query,
        "limit": limit,
        "output_directory": output_directory,
        "format": 'jpg',
        "chromedriver": "path/to/chromedriver.exe"  # Provide the path to your chromedriver executable
    }

    # Download images
    paths = response.download(arguments)

    return paths

if __name__ == "__main__":

    search_query = "rocket"

    # Number of images to download
    num_images = 10

    # Output directory for downloaded images
    output_folder = 'downloaded_images'

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Download images
    downloaded_paths = download_images(search_query, num_images, output_folder)

    print(f"Downloaded images saved in: {downloaded_paths[0][search_query]}")
