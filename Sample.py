import pandas as pd
import numpy as np
import os
import matplotlib as plt
import seaborne as sns
df=pd.read_csv("iris.csv")



import pandas as pd
df=pd.read_csv("C:\\Users\\Administrator\\Desktop\\iris.csv")
print(df)



import pandas as pd
df=pd.read_csv("C:\\Users\\Administrator\\Desktop\\iris.csv")
print(df.head)

import pandas as pd
df=pd.read_csv("C:\\Users\\Administrator\\Desktop\\iris.csv")
print(df.info)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("C:\\Users\\Administrator\\Desktop\\iris.csv")  
print (data.head(10))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("C:\\Users\\Administrator\\Desktop\\iris.csv")  
plt.plot(data["sepal.length"], "r--")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("C:\\Users\\Administrator\\Desktop\\iris.csv")  
plt.plot(data.Id,data["sepal.length"], "r--")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("C:\\Users\\Administrator\\Desktop\\iris.csv")  
data.plot(kind="scatter", x='sepal.length',y='petal.length')
plt.grid()

from sklearn import datasets
import matplotlib.pyplot as plt
bins = 20
iris = datasets.load_iris()
X_iris = iris.data
X_sepal = X_iris[:, 0]

plt.hist(X_sepal, bins)
plt.title("Histogram Sepal Length")
plt.xlabel(iris.feature_names[0])
plt.ylabel("Frequency")
plt.show
