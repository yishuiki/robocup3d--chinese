#!/bin/bash

# Call this script from any directory
# 从任何目录调用此脚本
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# cd to main folder
# 切换到主目录
cd "${SCRIPT_DIR}/.."

# 删除之前的打包文件夹
rm -rf ./bundle/build
rm -rf ./bundle/dist

# 定义 PyInstaller 的单文件选项
onefile="--onefile"

# 使用 PyInstaller 将应用程序、依赖项和数据文件打包为单个可执行文件
pyinstaller \
--add-data './world/commons/robots:world/commons/robots' \
--add-data './behaviors/slot/common:behaviors/slot/common' \
--add-data './behaviors/slot/r0:behaviors/slot/r0' \
--add-data './behaviors/slot/r1:behaviors/slot/r1' \
--add-data './behaviors/slot/r2:behaviors/slot/r2' \
--add-data './behaviors/slot/r3:behaviors/slot/r3' \
--add-data './behaviors/slot/r4:behaviors/slot/r4' \
--add-data './behaviors/custom/Dribble/*.pkl:behaviors/custom/Dribble' \
--add-data './behaviors/custom/Walk/*.pkl:behaviors/custom/Walk' \
--add-data './behaviors/custom/Fall/*.pkl:behaviors/custom/Fall' \
${onefile} --distpath ./bundle/dist/ --workpath ./bundle/build/ --noconfirm --name fcp Run_Player.py

# 创建启动脚本 start.sh
cat > ./bundle/dist/start.sh << EOF
#!/bin/bash
export OMP_NUM_THREADS=1

# 默认主机和端口
host=\${1:-localhost}
port=\${2:-3100}

# 启动 11 个球员
for i in {1..11}; do
  ./fcp -i \$host -p \$port -u \$i -t FCPortugal &
done
EOF

# 创建点球模式启动脚本 start_penalty.sh
cat > ./bundle/dist/start_penalty.sh << EOF
#!/bin/bash
export OMP_NUM_THREADS=1

# 默认主机和端口
host=\${1:-localhost}
port=\${2:-3100}

# 启动 1 号和 11 号球员（点球模式）
./fcp -i \$host -p \$port -u 1  -t FCPortugal -P 1 &
./fcp -i \$host -p \$port -u 11 -t FCPortugal -P 1 &
EOF

# 创建 Fat Proxy 模式启动脚本 start_fat_proxy.sh
cat > ./bundle/dist/start_fat_proxy.sh << EOF
#!/bin/bash
export OMP_NUM_THREADS=1

# 默认主机和端口
host=\${1:-localhost}
port=\${2:-3100}

# 启动 11 个球员（Fat Proxy 模式）
for i in {1..11}; do
  ./fcp -i \$host -p \$port -u \$i -t FCPortugal -F 1 &
done
EOF

# 创建杀进程脚本 kill.sh
cat > ./bundle/dist/kill.sh << EOF
#!/bin/bash
# 强制杀死所有名为 fcp 的进程
pkill -9 -e fcp
EOF

# 为生成的脚本文件添加执行权限
chmod a+x ./bundle/dist/start.sh
chmod a+x ./bundle/dist/start_penalty.sh
chmod a+x ./bundle/dist/start_fat_proxy.sh
chmod a+x ./bundle/dist/kill.sh