from resources import ConfigS3
define = ConfigS3()
folder_path = 'tomtom-voh'
for obj in define.bucket.objects.filter(Prefix=folder_path):
    print(obj.key)