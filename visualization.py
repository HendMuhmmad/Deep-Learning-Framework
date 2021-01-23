import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from PIL import Image
import numpy as np
import pandas as pd


class visualization:

    def __init__(self):
        style.use('seaborn')
        pass

    def view_photo_sample(self,sample,width,height,rgb):
        """
        :Description: 
        Displays an RGB or gray scale image from an input array and saves it in sample.jpg file.

        :parameter sample: input sample as an array of pixels.
        :type sample: array of float or int with image size and any shape.
        :parameter w: required image width.
        :type w: int.
        :parameter h: required image height.
        :type h: int.
        :parameter rgb: states wether the image is rgb or gray scale.
        :type rgb: bool.
        :returns:  none.
        """
        if rgb:
            data = (np.array(sample)*255).reshape(width,height,3)
            img = Image.fromarray(data.astype(np.uint8), 'RGB')
        else:
            data = (np.array(sample)*255).reshape(width,height)
            img = Image.fromarray(data.astype(np.uint8), 'L')

        img.show()
        img.save("sample.jpg")

    def __init(self):
      pass

        

    def __animate(self,i,values,xs,ys,title):
        """
        :Description: 
        A private helper method called by Funcanimate every time interval .

        :parameter i: frame number.
        :type i: int.
        :parameter values: values to be plotted.
        :type values: array of int.
        :parameter xs: empty array to be filled each frame with the frame number.
        :type xs: array of float.
        :parameter ys: empty array to be filled each frame from the values array with the coresponding value.
        :type ys: array of float.
        :parameter title: figure title.
        :type title: str.
        :returns: none.
        """
        xs.append(i)
        ys.append(values[i])
        plt.cla()
        plt.title(title)
        plt.xlabel("iteration")
        plt.ylabel(title)
        plt.plot(xs,ys,color='purple')
        
        

    def live_visualization(self,values):
        """
        :Description: 
        Plots a live visualization for the input vs the number of iterations.

        :parameter values: input values to be graphed.
        :type values: dictionary whose key is the label(str)  and whose items are the values(float) of that label.
        :returns: none.
        """
        title = list(values.keys())[0]
        size = len(values[title])
        xs = []
        ys = []
        ani = animation.FuncAnimation(plt.gcf(), self.__animate,init_func=self.__init, interval=1000,frames=size,fargs=(values[title],xs,ys,title),repeat=False).save('animation.gif', writer='pillow')
        plt.tight_layout()
        plt.show()
        


  