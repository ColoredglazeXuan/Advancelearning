Windows下WSL2环境搭建：

##### wsl安装：

​		powershell使用管理员身份打开，然后输入以下命令：

```shell
wsl --install
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
wsl --set-default-version 2
```

然后重启电脑。

##### Ubuntu装载到D盘

​		进入到装载Ubuntu的文件夹中，用以下命令下载Ubuntu：

```shell
Invoke-WebRequest -Uri https://wsldownload.azureedge.net/Ubuntu_2004.2020.424.0_x64.appx -OutFile Ubuntu20.04.appx -UseBasicParsing
```

​		然后进行装载：

```shell
Rename-Item .\Ubuntu20.04.appx Ubuntu.zip
Expand-Archive .\Ubuntu.zip -Verbose
cd .\Ubuntu\
.\ubuntu2004.exe
```

​	（ubuntu命名时存在一定规则，好像是不能用常用人名（英文的）命名？开始使用的Steve，命名失败）

##### 文件操作命令

```python
pwd 		#打出当前工作目录名
cd			#跳转
ls 			#列出目录内容
less		#浏览文件
cp			#复制文件和目录 cp item1 item2，从1复制到2
mv			#移动/重命名文件和目录 mv item1 item2，从1移动到2，并且改名为2中文件名
mkdir		#创建目录，可一次输入多个文件夹名，空格隔开，从而创建多个文件夹
rm			#删除文件和目录，加入“-f”可以强制删除
```

##### 辅助命令

```python
type 		#显示命令类别，以及是否是其他命令的别名
which		#显示命令的程序位置（不包括内部命令和命令别名）
man			#显示命令的使用手册，部分命令存在使用手册，包含命令语法的纲要、可选项和说明等


alias		#创建命令别名，alias name = 'string'，string包含需要执行的一系列命令，相当于顺序执行字符串中的命令
--help		#很多程序有的一个可选项，用于显示命令语法和说明
```

