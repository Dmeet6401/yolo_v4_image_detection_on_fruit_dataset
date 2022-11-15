# yolo_v4_image_detection_on_fruit_dataset
## Changes needs to be done before training model
after downloading the darknet check .cfg (CONFIG) file in which do changes as mentioned below
1) Change batch to 64 and subdivisiont to 16(if not works than 64)
2) Change height and width (gigher the value higher the accuracy but more time to exicute)
3) Change max_batches (less than 3 classes it would be 6000) (for more class = no. class * 2000)
4) Change Steps (Steps = (90% of max_batch, 80% of max_batch))
5) Search yolo find filter above yolo and change it ((classes + 5)*3)
6) Below yolo find class and change it to your class
