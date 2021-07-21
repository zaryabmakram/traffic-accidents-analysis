import matplotlib.pyplot as plt

def plot_loss_curve(history, legend=['Train', 'Validation']):
    # plot loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(legend, loc='upper left')
    plt.savefig('assets/loss_plot.png')   # saving as .png
    plt.show()