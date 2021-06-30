import utils as ut
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
path="./"
data=ut.importDatainfo(path)

data=ut.balanceData(data,False) 

imagePath,Steering=ut.LoadData(data)
# print(imagePath[0],Steering[0])

X,Xval,Y,Yval=train_test_split(imagePath,Steering,test_size=0.2,random_state=5)
# print(len(X),len(Y))
# print(len(Xval),len(Yval))

model=ut.Model()
model.summary() 

history=model.fit(ut.batchGen(X,Y,200,True),steps_per_epoch=300, epochs=20,validation_data=ut.batchGen(Xval,Yval,200,False),validation_steps=200)
#training---30000 image per epochs   validation----20000 image per epochs for 10 epochs
model.save('model.h5' ) #h5  saves the weight and the architecture
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training','Validation'])
plt.ylim([0,1])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()

