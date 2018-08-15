#!/usr/bin/env python3

import keras
import matplotlib.pylab as plt
from numpy import sqrt

class liveHist(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.val_rmse_acc = []
        self.val_mae_acc = []
        self.rmse_acc = []
        self.mae_acc = []
        self.losses = []
        self.val_losses = []

        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, sharex=True)
        self.fig.text(.5, .04, 'Epochs', ha='center')


    def on_epoch_end(self, epoch, logs={}):
        self.val_rmse_acc.append(sqrt(logs.get('val_mean_squared_error')))
        self.val_mae_acc.append(logs.get('val_mean_absolute_error'))
        self.rmse_acc.append(sqrt(logs.get('mean_squared_error')))
        self.mae_acc.append(logs.get('mean_absolute_error'))
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.x.append(self.i)
        self.i += 1

        self.ax1.cla()
        self.ax2.cla()
        self.ax3.cla()
        self.ax1.set_title('Losses')
        self.ax1.set(ylabel='Logarithmic MSE')

        self.ax2.set_title('Accuracy (RMSE)')
        # self.ax2.set(ylabel='Mean Squared Error')

        self.ax3.set_title('Accuracy (MAE)')
        # self.ax3.set(ylabel='Mean Absolute Error')

        self.ax1.plot(self.x, self.losses, label="loss", color='C0')
        self.ax1.plot(self.x, self.val_losses, label="val_loss", color='C1')
        self.ax1.legend(loc="upper right")
        self.ax2.plot(self.x, self.rmse_acc, label="training", color='C0')
        self.ax2.plot(self.x, self.val_rmse_acc, label="validation", color='C1')
        self.ax2.legend(loc="upper right")
        self.ax3.plot(self.x, self.mae_acc, label="training", color='C0')
        self.ax3.plot(self.x, self.val_mae_acc, label="validation", color='C1')
        self.ax3.legend(loc="upper right")

        plt.pause(.01)
        plt.draw()
