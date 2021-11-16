import pandas as pd
import numpy as np
import PIL.Image
from skimage import filters
from skimage import feature
import cv2

class get_data(object):

    def constant(self):
        non_dem = []
        for i in range(0,2560):
            img = PIL.Image.open('nonDem' + str(i) + '.jpg')
            data = np.asarray(img)
            if len(data.flatten()) == 36608:
                x = data.flatten()
                y = x.tolist()
                non_dem.append(y)
        ar = np.array(non_dem)
        df = pd.DataFrame(ar)

        tar = []

        for _ in range(0, 2560):
            tar.append(0)

        df['target'] = tar

        ## Get data for demented observations
        dem = []

        for i in range(0, 1792):
            img = PIL.Image.open('verymildDem' + str(i) + '.jpg')
            data = np.asarray(img)
            if len(data.flatten()) == 36608:
                x = data.flatten()
                y = x.tolist()
                dem.append(y)
        for i in range(0, 44):
            img = PIL.Image.open('moderateDem' + str(i) + '.jpg')
            data = np.asarray(img)
            if len(data.flatten()) == 36608:
                x = data.flatten()
                y = x.tolist()
                dem.append(y)
        for i in range(0, 716):
            img = PIL.Image.open('mildDem' + str(i) + '.jpg')
            data = np.asarray(img)
            if len(data.flatten()) == 36608:
                x = data.flatten()
                y = x.tolist()
                dem.append(y)

        ar = np.array(dem)
        df2 = pd.DataFrame(ar)

        tar2 = []

        for _ in range(0, 2552):
            tar2.append(1)

        df2['target'] = tar2

        final = pd.concat([df, df2])
        final = final.reset_index(drop = True)
        
        return final

    def roberts(self):
        non_dem = []
        for i in range(0,2560):
            img = PIL.Image.open('nonDem' + str(i) + '.jpg')
            edge_roberts = filters.roberts(img) * 1000
            data = np.asarray(edge_roberts)
            if len(data.flatten()) == 36608:
                x = data.flatten()
                y = x.tolist()
                non_dem.append(y)
        ar = np.array(non_dem)
        df = pd.DataFrame(ar)

        tar = []

        for _ in range(0, 2560):
            tar.append(0)

        df['target'] = tar

        ## Get data for demented observations
        dem = []

        for i in range(0, 1792):
            img = PIL.Image.open('verymildDem' + str(i) + '.jpg')
            edge_roberts = filters.roberts(img) * 1000
            data = np.asarray(edge_roberts)
            if len(data.flatten()) == 36608:
                x = data.flatten()
                y = x.tolist()
                dem.append(y)
        for i in range(0, 44):
            img = PIL.Image.open('moderateDem' + str(i) + '.jpg')
            edge_roberts = filters.roberts(img) * 1000
            data = np.asarray(edge_roberts)
            if len(data.flatten()) == 36608:
                x = data.flatten()
                y = x.tolist()
                dem.append(y)
        for i in range(0, 716):
            img = PIL.Image.open('mildDem' + str(i) + '.jpg')
            edge_roberts = filters.roberts(img) * 1000
            data = np.asarray(edge_roberts)
            if len(data.flatten()) == 36608:
                x = data.flatten()
                y = x.tolist()
                dem.append(y)

        ar = np.array(dem)
        df2 = pd.DataFrame(ar)

        tar2 = []

        for _ in range(0, 2552):
            tar2.append(1)

        df2['target'] = tar2

        final = pd.concat([df, df2])
        final = final.reset_index(drop = True)
        
        return final

    def sobel(self):
        non_dem = []
        for i in range(0,2560):
            img = PIL.Image.open('nonDem' + str(i) + '.jpg')
            edge_sobel = filters.sobel(img) * 1000
            data = np.asarray(edge_sobel)
            if len(data.flatten()) == 36608:
                x = data.flatten()
                y = x.tolist()
                non_dem.append(y)
        ar = np.array(non_dem)
        df = pd.DataFrame(ar)

        tar = []

        for _ in range(0, 2560):
            tar.append(0)

        df['target'] = tar

        ## Get data for demented observations
        dem = []

        for i in range(0, 1792):
            img = PIL.Image.open('verymildDem' + str(i) + '.jpg')
            edge_sobel = filters.sobel(img) * 1000
            data = np.asarray(edge_sobel)
            if len(data.flatten()) == 36608:
                x = data.flatten()
                y = x.tolist()
                dem.append(y)
        for i in range(0, 44):
            img = PIL.Image.open('moderateDem' + str(i) + '.jpg')
            edge_sobel = filters.sobel(img) * 1000
            data = np.asarray(edge_sobel)
            if len(data.flatten()) == 36608:
                x = data.flatten()
                y = x.tolist()
                dem.append(y)
        for i in range(0, 716):
            img = PIL.Image.open('mildDem' + str(i) + '.jpg')
            edge_sobel = filters.sobel(img) * 1000
            data = np.asarray(edge_sobel)
            if len(data.flatten()) == 36608:
                x = data.flatten()
                y = x.tolist()
                dem.append(y)

        ar = np.array(dem)
        df2 = pd.DataFrame(ar)

        tar2 = []

        for _ in range(0, 2552):
            tar2.append(1)

        df2['target'] = tar2

        final = pd.concat([df, df2])
        final = final.reset_index(drop = True)
        
        return final

    def prewitt(self):
        non_dem = []
        for i in range(0,2560):
            img = PIL.Image.open('nonDem' + str(i) + '.jpg')
            edge_prewitt = filters.prewitt(img) * 1000
            data = np.asarray(edge_prewitt)
            if len(data.flatten()) == 36608:
                x = data.flatten()
                y = x.tolist()
                non_dem.append(y)
        ar = np.array(non_dem)
        df = pd.DataFrame(ar)

        tar = []

        for _ in range(0, 2560):
            tar.append(0)

        df['target'] = tar

        ## Get data for demented observations
        dem = []

        for i in range(0, 1792):
            img = PIL.Image.open('verymildDem' + str(i) + '.jpg')
            edge_prewitt = filters.prewitt(img) * 1000
            data = np.asarray(edge_prewitt)
            if len(data.flatten()) == 36608:
                x = data.flatten()
                y = x.tolist()
                dem.append(y)
        for i in range(0, 44):
            img = PIL.Image.open('moderateDem' + str(i) + '.jpg')
            edge_prewitt = filters.prewitt(img) * 1000
            data = np.asarray(edge_prewitt)
            if len(data.flatten()) == 36608:
                x = data.flatten()
                y = x.tolist()
                dem.append(y)
        for i in range(0, 716):
            img = PIL.Image.open('mildDem' + str(i) + '.jpg')
            edge_prewitt = filters.prewitt(img) * 1000
            data = np.asarray(edge_prewitt)
            if len(data.flatten()) == 36608:
                x = data.flatten()
                y = x.tolist()
                dem.append(y)

        ar = np.array(dem)
        df2 = pd.DataFrame(ar)

        tar2 = []

        for _ in range(0, 2552):
            tar2.append(1)

        df2['target'] = tar2

        final = pd.concat([df, df2])
        final = final.reset_index(drop = True)
        
        return final

    def canny(self):
        non_dem = []
        for i in range(0,2560):
            img = cv2.imread('nonDem' + str(i) + '.jpg', cv2.IMREAD_GRAYSCALE)
            #img = PIL.Image.open('nonDem' + str(i) + '.jpg')
            edge_canny = feature.canny(img, sigma = 3) * 1000
            data = np.asarray(edge_canny)
            if len(data.flatten()) == 36608:
                x = data.flatten()
                y = x.tolist()
                non_dem.append(y)
        ar = np.array(non_dem)
        df = pd.DataFrame(ar)

        tar = []

        for _ in range(0, 2560):
            tar.append(0)

        df['target'] = tar

        ## Get data for demented observations
        dem = []

        for i in range(0, 1792):
            img = cv2.imread('verymildDem' + str(i) + '.jpg', cv2.IMREAD_GRAYSCALE)
            #img = PIL.Image.open('verymildDem' + str(i) + '.jpg')
            edge_canny = feature.canny(img, sigma = 3) * 1000
            data = np.asarray(edge_canny)
            if len(data.flatten()) == 36608:
                x = data.flatten()
                y = x.tolist()
                dem.append(y)
        for i in range(0, 44):
            img = cv2.imread('moderateDem' + str(i) + '.jpg', cv2.IMREAD_GRAYSCALE)
            #img = PIL.Image.open('moderateDem' + str(i) + '.jpg')
            edge_canny = feature.canny(img, sigma = 3) * 1000
            data = np.asarray(edge_canny)
            if len(data.flatten()) == 36608:
                x = data.flatten()
                y = x.tolist()
                dem.append(y)
        for i in range(0, 716):
            img = cv2.imread('mildDem' + str(i) + '.jpg', cv2.IMREAD_GRAYSCALE)
            #img = PIL.Image.open('mildDem' + str(i) + '.jpg')
            edge_canny = feature.canny(img, sigma = 3) * 1000
            data = np.asarray(edge_canny)
            if len(data.flatten()) == 36608:
                x = data.flatten()
                y = x.tolist()
                dem.append(y)

        ar = np.array(dem)
        df2 = pd.DataFrame(ar)

        tar2 = []

        for _ in range(0, 2552):
            tar2.append(1)

        df2['target'] = tar2

        final = pd.concat([df, df2])
        final = final.reset_index(drop = True)
        
        return final