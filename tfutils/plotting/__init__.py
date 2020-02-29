class DynamicPlot:
    def __init__(self, plt, rows, cols, figsize):
        
        self.rows = rows
        self.cols = cols

        self.plt = plt 

        self.fig = self.plt.figure(figsize=figsize)
        self.plt.ion()
    
    def draw_fig(self):
        self.fig.show()
        self.fig.canvas.draw()
    
    def start_of_epoch(self, epoch):
        self.ax = self.plt.subplot(self.rows, self.cols, 1 + epoch)
        
    def end_of_epoch(self, image, cmap, xlabel, ylabel):
        self.ax.imshow(image, cmap=cmap)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.draw_fig()