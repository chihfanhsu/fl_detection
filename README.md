# Facial landmarks detection

1. Simple cnn <br/>
training: <br/>
python training_reg.py --gpu=0 --tar_model=reg_1 --batch_size=16 --training=True --testing=0 --lr=0.001 <br/>
testing: <br/>
python training_reg.py --gpu=1 --tar_model=reg_1 --batch_size=16 --testing=0 <br/>

2. Deep Alignment Network <br/>
training: <br/>
python training_reg.py --gpu=0 --tar_model=dan --batch_size=16 --training=True --testing=0 --lr=0.001 <br/>
testing: <br/>
python training_reg.py --gpu=0 --tar_model=dan --batch_size=16 --testing=0 <br/>
