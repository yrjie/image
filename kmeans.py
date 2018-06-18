import sys, os
import numpy as np                
import PIL.Image as image         
from sklearn.cluster import KMeans

if len(sys.argv) < 2:
    print("Usage: filename")
    exit(1)

infile = sys.argv[1]
outfile = "outfile/" + os.path.basename(infile)
 
def loadData(filePath):
    f = open(filePath,'rb')       
    data = []
    img = image.open(f)
    m,n = img.size     
    if m>800:
        img = img.resize((800, 800*n/m))
        m,n = img.size
    for i in range(m):
        for j in range(n):
            x,y,z = img.getpixel((i,j))
            data.append([x/255.0,y/255.0,z/255.0])
    f.close()
    return np.mat(data),m,n
 
ext = 100
color_size = 50
marg_size = 25
cnum = 4

imgData,row,col = loadData(infile)
km = KMeans(n_clusters=cnum).fit(imgData)
 
label = km.labels_.reshape([row,col])
# print(km.cluster_centers_)

pic_new = image.new("RGB", (row + ext, col))
for i in range(ext):
    for j in range(col):
        pic_new.putpixel((row+i, j), (255, 255, 255))

for k in range(cnum):
    color = tuple([int(x*255) for x in km.cluster_centers_[k]])
    left = row+marg_size
    top = (k+1)*marg_size + k*color_size
    for i in range(color_size):
        for j in range(color_size):
            pic_new.putpixel((left+i, top+j), color)

for i in range(row):
    for j in range(col):
        arr = np.asarray(imgData[i*col+j])
        color = tuple([int(x*255) for x in arr[0]])
        pic_new.putpixel((i,j), color)
        # pic_new.putpixel((i,j), int(255/(label[i][j]+1)))
pic_new.save(outfile, "JPEG")
