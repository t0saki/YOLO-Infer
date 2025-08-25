from ultralytics import YOLO, checks, hub
checks()

hub.login('6f8b2bacaa1a1075454f7a02e40714dca180232d01')

model = YOLO('https://hub.ultralytics.com/models/p6gpPI5oNp5IT6KwzgID')
results = model.train()
session = model.session
model.load("/Users/tosaki/dev/YOLO-Infer/yolo11n.pt")

ret = session.upload_model(is_best = True, epoch = 100,
                         weights ="/Users/tosaki/dev/YOLO-Infer/yolo11n.pt", final=True)

print(ret)