from ultralytics import settings

# View all settings
print(settings, "\n")

# View datasets_dir setting
print("before: " + settings["datasets_dir"])

# Update the datasets_dir setting
settings.update({"datasets_dir": "/mnt/d/LFDev-D"})

# View datasets_dir setting after updating
print("after: " + settings["datasets_dir"])