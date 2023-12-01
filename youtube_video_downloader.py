import pytube

# youtube video link
youtube_link = "YOUTUBE-VIDEO-LINK"

# location to download video
location = "LOCATION-TO-DOWNLOAD"

pytube.YouTube(youtube_link).streams.filter(progressive=True, file_extension='mp4').order_by('resolution').last().download(location)