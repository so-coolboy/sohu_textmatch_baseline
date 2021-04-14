# sohu_textmatch_baseline
这个baseline使用transformers搭的最简单的baseline，线上分数是65左右。

# 简单思路
使用一个中文bert先预测A类，再预测B类，config.py中有一个flag参数设定是预测A类还是B类。

# 运行方法
由于我一般是在vs code这种IDE中运行，所以没有写命令行参数，感兴趣的同学只需要先看一下preprocess.py文件，
修改好路径，之后会在input文件夹下生成A类，B类分别对应的train，valid和test，然后运行train.py就可以了。

评价函数这里用的acc，大家自己改成f1，说不定效果会好点。

写出来只是给初学者提供一点帮助，自己也获得过很多人的帮助。有问题请大家指出。
