from hub_sdk import HUBClient

credentials = {"api_key": "6f8b2bacaa1a1075454f7a02e40714dca180232d01"}
client = HUBClient(credentials)

# Fetches the first page with 10 models
# model_list = client.model_list(page_size=10)
# # Displays the current page's models
# print("Current result:", model_list.results)

# model_list.next()  # Move to the next page
# print("Next page result:", model_list.results)

# model_list.previous()  # Return to the previous page
# print("Previous page result:", model_list.results)

model = client.model("p6gpPI5oNp5IT6KwzgID")
weight_url = model.get_weights_url("best")  # or "last"
print("Weight URL link:", weight_url)


# Uploads the specified model checkpoint
ret = model.upload_model(is_best=True, epoch=100,
                         weights="/Users/tosaki/dev/YOLO-Infer/yolo11n.pt", final=True)

print("Upload result:", ret)