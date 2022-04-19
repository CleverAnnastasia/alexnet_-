复现Alexnet五层卷积，三层全连接。
引入RELU函数；DROPOUT去掉部分神经节点，防止过拟合；层叠池化操作以往池化的大小PoolingSize与步长stride一般是相等的，例如：图像大小为256*256，PoolingSize=2×2，stride=2，实现结果如图：
![f613a2ce3badc780ebde23c044f558b](https://user-images.githubusercontent.com/44082531/164039174-a60b06f5-a48f-4617-8149-298e8f54ce68.png)

![553762eef088159fefcd370c4f5fba3](https://user-images.githubusercontent.com/44082531/164039131-d9b773a8-57a3-4af7-9ae9-7fc28761ea0d.png)
