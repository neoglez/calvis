#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:05:21 2020

@author: neoglez
"""

from fastai.vision import *
###############################################################################
# We annotate the CMU dataset with human dimensions
projectDir = "/home/neoglez/cmu/"
rootDir = "/home/neoglez/cmu/dataset/"
imgDir = "/home/neoglez/cmu/dataset/synthetic_images/200x200/"
###############################################################################

# subclassing ItemBase
class CalvisImage(ItemBase):
    def __init__(self, img1):
        self.img1 = img1
        self.obj, self.data = img1,[img1.data]
        
    def apply_tfms(self, tfms, **kwargs):
        self.img1 = self.img1.apply_tfms(tfms, **kwargs)
        self.data = [self.img1.data]
        return self
    
    def to_one(self): return Image(torch.cat(self.data))

# subclassing ItemList
class CalvisTargetList(ItemList):
    def reconstruct(self, t:Tensor): 
        if len(t.size()) == 0: return t
        return Image(t[0])
    
class CalvisImageList(ImageList):
    _label_cls=CalvisTargetList
    
    def __init__(self, items, itemsB=None, **kwargs):
        super().__init__(items, **kwargs)
        self.itemsB = itemsB
        self.copy_new.append('itemsB')
        
    def get(self, i):
        img1 = super().get(i)
        fn = self.itemsB[random.randint(0, len(self.itemsB)-1)]
        return Image(img1, open_image(fn))
    
    @classmethod
    def from_directory(cls, path,**kwargs):
        itemsB = ImageList.from_folder(path/folderB).items
        res = super().from_folder(path/folderA, itemsB=itemsB, **kwargs)
        res.path = path
        return res
    
    def reconstruct(self, t:Tensor): 
        return Image(t[0])
    
    def show_xys(self, xs, ys, figsize:Tuple[int,int]=(12,6), **kwargs):
        '''Show the `xs` and `ys` on a figure of `figsize`.
        `kwargs` are passed to the show method.'''
        rows = int(math.sqrt(len(xs)))
        fig, axs = plt.subplots(rows,rows,figsize=figsize)
        for i, ax in enumerate(axs.flatten() if rows > 1 else [axs]):
            xs[i].to_one().show(ax=ax, **kwargs)
        plt.tight_layout()

    def show_xyzs(self, xs, ys, zs, figsize:Tuple[int,int]=None, **kwargs):
        """Show `xs` (inputs), `ys` (targets) and `zs` (predictions) on a 
        figure of `figsize`.
        `kwargs` are passed to the show method."""
        figsize = ifnone(figsize, (12,3*len(xs)))
        fig,axs = plt.subplots(len(xs), 2, figsize=figsize)
        fig.suptitle('Ground truth / Predictions', weight='bold', size=14)
        for i,(x,z) in enumerate(zip(xs,zs)):
            x.to_one().show(ax=axs[i,0], **kwargs)
            z.to_one().show(ax=axs[i,1], **kwargs)





# Let's begin with our sample of the CMU dataset.

#calvis_cmu = untar_data(URLs.MNIST_TINY)
tfms = get_transforms(do_flip=False)

data = (CalvisImageList.from_directory(rootDir)
        .transform(tfms)
        .databunch()) 



