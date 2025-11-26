# 足球社交网站

一个集技巧分享、约球功能和视频分析于一体的足球社交平台。

## 功能特色

### 🎯 核心功能
- **技巧分享**: 上传和观看足球技巧视频
- **约球系统**: 组织和参加足球比赛
- **视频分析**: AI驱动的比赛视频分析
- **数据统计**: 个人和团队数据统计
- **排行榜**: 多维度排名系统

### 🤖 AI视频分析
- 球员动作识别
- 比赛数据自动统计
- 技术动作分析
- 战术分析

### 📱 社交功能
- 用户关注系统
- 视频评论和点赞
- 比赛分享
- 个人主页

## 技术栈

### 后端
- Node.js + Express.js
- MongoDB
- JWT认证
- Socket.io实时通信

### 前端
- React 18
- Bootstrap + Material Design
- Chart.js数据可视化
- React Router

### AI分析
- Python + FastAPI
- OpenCV视频处理
- 多模态AI模型
- 机器学习算法

### 部署
- Docker容器化
- Docker Compose编排
- Nginx反向代理

## 快速开始

### 环境要求
- Docker & Docker Compose
- Node.js 18+ (本地开发)
- Python 3.9+ (本地开发)

### 使用Docker启动

1. 克隆项目
```bash
git clone <repository-url>
cd football-social
```

2. 配置环境变量
```bash
cp backend/.env.example backend/.env
# 编辑 .env 文件，配置数据库连接等
```

3. 启动服务
```bash
docker-compose up -d
```

4. 访问应用
- 前端: http://localhost:3000
- 后端API: http://localhost:5000
- 视频分析API: http://localhost:6000

### 本地开发

#### 后端开发
```bash
cd backend
npm install
npm run dev
```

#### 前端开发
```bash
cd frontend
npm install
npm start
```

#### 视频分析服务
```bash
cd video-analysis
pip install -r requirements.txt
python main.py
```

## API文档

### 认证相关
- `POST /api/auth/register` - 用户注册
- `POST /api/auth/login` - 用户登录
- `GET /api/auth/me` - 获取当前用户信息

### 视频相关
- `GET /api/videos` - 获取视频列表
- `POST /api/videos` - 上传视频
- `GET /api/videos/:id` - 获取视频详情

### 比赛相关
- `GET /api/matches` - 获取比赛列表
- `POST /api/matches` - 创建比赛
- `GET /api/matches/:id` - 获取比赛详情

### 视频分析
- `POST /analyze` - 分析视频
- `GET /analysis/:filename` - 获取分析结果

## 项目结构

```
football-social/
├── backend/              # Node.js后端
│   ├── models/          # 数据模型
│   ├── routes/          # API路由
│   ├── middleware/      # 中间件
│   ├── controllers/     # 控制器
│   └── utils/           # 工具函数
├── frontend/            # React前端
│   ├── src/
│   │   ├── components/  # 组件
│   │   ├── pages/       # 页面
│   │   ├── services/    # API服务
│   │   └── utils/       # 工具函数
├── video-analysis/      # Python视频分析
│   ├── main.py         # FastAPI应用
│   └── analyzers/      # 分析器模块
├── database/           # 数据库脚本
└── docker-compose.yml  # 容器编排
```

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交Issue或联系开发团队。