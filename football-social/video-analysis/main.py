from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import tempfile
from pathlib import Path
import cv2
import numpy as np
from typing import Dict, Any
import json

app = FastAPI(title="足球视频分析API", version="1.0.0")

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 上传目录
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

class FootballVideoAnalyzer:
    def __init__(self):
        self.ball_detector = self._init_ball_detector()
        self.player_detector = self._init_player_detector()
        self.action_classifier = self._init_action_classifier()
    
    def _init_ball_detector(self):
        """初始化足球检测器"""
        # 这里可以加载预训练的足球检测模型
        return None
    
    def _init_player_detector(self):
        """初始化球员检测器"""
        # 这里可以加载预训练的球员检测模型
        return None
    
    def _init_action_classifier(self):
        """初始化动作分类器"""
        # 这里可以加载预训练的动作分类模型
        return None
    
    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """分析足球视频"""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            # 模拟分析结果
            analysis_data = {
                "duration": duration,
                "fps": fps,
                "total_frames": total_frames,
                "player_actions": self._detect_player_actions(cap, fps),
                "match_stats": self._calculate_match_stats(cap, fps),
                "techniques": self._analyze_techniques(cap, fps)
            }
            
            cap.release()
            return analysis_data
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"视频分析失败: {str(e)}")
    
    def _detect_player_actions(self, cap, fps):
        """检测球员动作"""
        actions = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 每秒分析一次
            if frame_count % int(fps) == 0:
                # 模拟动作检测
                if frame_count % (int(fps) * 5) == 0:  # 每5秒一个动作
                    actions.append({
                        "action": "射门",
                        "timestamp": frame_count / fps,
                        "confidence": 0.85,
                        "position": {"x": 320, "y": 240}
                    })
            
            frame_count += 1
        
        return actions
    
    def _calculate_match_stats(self, cap, fps):
        """计算比赛统计数据"""
        return {
            "totalPasses": np.random.randint(50, 200),
            "successfulPasses": np.random.randint(30, 150),
            "shots": np.random.randint(5, 20),
            "goals": np.random.randint(0, 5),
            "assists": np.random.randint(0, 5),
            "distance": round(np.random.uniform(5.0, 12.0), 1)
        }
    
    def _analyze_techniques(self, cap, fps):
        """分析技术动作"""
        techniques = [
            {"name": "传球", "count": np.random.randint(20, 50), "successRate": round(np.random.uniform(0.6, 0.9), 2)},
            {"name": "射门", "count": np.random.randint(3, 10), "successRate": round(np.random.uniform(0.2, 0.4), 2)},
            {"name": "盘带", "count": np.random.randint(10, 30), "successRate": round(np.random.uniform(0.7, 0.9), 2)},
            {"name": "头球", "count": np.random.randint(2, 8), "successRate": round(np.random.uniform(0.5, 0.8), 2)}
        ]
        return techniques

analyzer = FootballVideoAnalyzer()

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    """分析上传的足球视频"""
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="请上传视频文件")
    
    # 保存上传的文件
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        # 分析视频
        analysis_result = analyzer.analyze_video(tmp_file_path)
        
        # 保存分析结果
        result_path = UPLOAD_DIR / f"{file.filename.split('.')[0]}_analysis.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, ensure_ascii=False, indent=2)
        
        return {
            "message": "视频分析完成",
            "analysis": analysis_result,
            "result_file": str(result_path)
        }
    
    finally:
        # 清理临时文件
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

@app.get("/analysis/{filename}")
async def get_analysis_result(filename: str):
    """获取分析结果"""
    result_path = UPLOAD_DIR / filename
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="分析结果不存在")
    
    with open(result_path, 'r', encoding='utf-8') as f:
        analysis_data = json.load(f)
    
    return analysis_data

@app.get("/")
async def root():
    return {"message": "足球视频分析API运行中"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6000)