# EDSRNet-GPU--GameSRNet
EDSRNet-GPU : A super-resolution net for game map rebuild
(1)prepare your data, put imgs into ./data_dir.
(2)python2 cut_img.py (it will create "./data_dir_crop_y" and put img-patches of 100*100 into it.)
(3)python2 create_blur_img.py (it will create "./data_dir_crop_x" and put img-patches-resize of 50*50 into it.)
(4)python2 train.py --dataset data_dir (begin train)
(5)put your test-imgs into ./data
(6)python2 test.py --image a.jpg --savedir saved_models (it will test all imgs in "./data")
