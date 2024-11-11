import super_gradients

yolo_nas = super_gradients.training.models.get("yolo_nas_pose_l", pretrained_weights="coco_pose").cuda()
model_predictions  = yolo_nas.predict("https://deci-pretrained-models.s3.amazonaws.com/sample_images/beatles-abbeyroad.jpg", conf=0.5).show()

prediction = model_predictions[0].prediction # One prediction per image - Here we work with 1 image, so we get the first.

bboxes = prediction.bboxes_xyxy # [Num Instances, 4] List of predicted bounding boxes for each object 
poses  = prediction.poses       # [Num Instances, Num Joints, 3] list of predicted joints for each detected object (x,y, confidence)
scores = prediction.scores      # [Num Instances] - Confidence value for each predicted instance