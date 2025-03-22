import yt_dlp
import os
import json
from urllib.parse import urlparse, parse_qs

def get_video_id(url):
    """Extract YouTube video ID from URL."""
    parsed = urlparse(url)
    if parsed.hostname in ['youtu.be']:
        return parsed.path[1:]
    elif parsed.hostname in ['www.youtube.com', 'youtube.com']:
        return parse_qs(parsed.query).get('v', [None])[0]
    return None

def download_youtube_audio(url_list, output_folder="podcasts"):
    os.makedirs(output_folder, exist_ok=True)

    # Remove duplicates by video ID
    seen = set()
    unique_urls = []
    for url in url_list:
        vid = get_video_id(url)
        if vid and vid not in seen:
            seen.add(vid)
            unique_urls.append(url)

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_folder, '%(title).70s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': False,
        'noplaylist': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for url in unique_urls:
            try:
                print(f"⬇️  Downloading: {url}")
                info_dict = ydl.extract_info(url, download=True)

                # Save metadata
                metadata = {
                    "title": info_dict.get("title"),
                    "upload_date": info_dict.get("upload_date"),  # format: YYYYMMDD
                    "uploader": info_dict.get("uploader"),
                    "webpage_url": info_dict.get("webpage_url"),
                }

                # Convert upload date to readable format
                if metadata["upload_date"]:
                    d = metadata["upload_date"]
                    metadata["upload_date"] = f"{d[:4]}-{d[4:6]}-{d[6:]}"

                # Save metadata JSON
                base_name = info_dict.get("title", "unknown_title")[:70].strip()
                json_path = os.path.join(output_folder, f"{base_name}.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)

                print(f"✅ Saved: {base_name}.mp3 + metadata")
            except Exception as e:
                print(f"❌ Failed to download {url}: {e}")

if __name__ == "__main__":
    youtube_links = [
        # your full list of links here
    ]
    download_youtube_audio(youtube_links)
