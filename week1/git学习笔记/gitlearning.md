git工具学习



git仓库建立

```python
git init #将当前目录变成git管理的仓库
```

文件添加和修改等

```python
git add readme.txt #将readme.txt文件添加到暂存区
git commit -m '备注' #将放在暂存区的文件全部提交到仓库
git diff file #查看文件修改内容

git log #查看每一次commit的历史记录
git reset --hard HEAD #将当前仓库的文件回退到最近一次commit的版本，增加'^'符号来增加回退的版本，也可以直接把HEAD换成版本号
git checkout -- file #丢弃工作区的修改

git rm file #删除文件
```

文件修改状况、暂存区文件添加情况查看

```python
git status
```

远程仓库连接

​	连接本地git仓库和github仓库首先需要设置SSH连接，使用以下命令创建本地的SSH Key，文件生成在C:\Users\用户名\\.ssh，有id_rsa和id_rsa.pub两个文件，其中id_rsa.pub是公钥，需要在github账户中设置SSH Key时添加进去。

```
ssh-keygen -t rsa –C “youremail@example.com”
```

github和本地仓库的SSH连接设置好后，创建一个git仓库，使用以下命令即可将本地仓库的内容与github上的git仓库相关联。

```python
git remote add origin https://github.com/ColoredglazeXuan/Advancelearning.git
```

然后使用git push命令推送到github上。

```
git push -u origin master #master表示主分支，加上-u会将本地主分支和github上的主分支关联到一起，以后就不用加-u了
```





### 分支管理

```python
git checkout -b dev #创建出dev分支，加了-b会在创建时直接切换到到新分支上

git branch #查看分支和当前所处分支

git branch name #创建分支

git checkout name #切换分支

git merge name #合并目标分支到当前分支

git branch -d name #删除分支
```

#### 部分问题

```python
git config --global core.autocrlf true #解决'LF will be replaced by CRLF the next time Git touches it'问题，Dos/Windows平台默认换行符：回车（CR）+换行（LF），即’\r\n’，Mac/Linux平台默认换行符：换行（LF），即’\n’
```

