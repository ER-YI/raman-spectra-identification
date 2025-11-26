const mongoose = require('mongoose');

const videoSchema = new mongoose.Schema({
  title: {
    type: String,
    required: true,
    trim: true,
    maxlength: 100
  },
  description: {
    type: String,
    maxlength: 1000,
    trim: true
  },
  uploadedBy: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  videoUrl: {
    type: String,
    required: true
  },
  thumbnailUrl: {
    type: String,
    default: ''
  },
  duration: {
    type: Number,
    default: 0
  },
  fileSize: {
    type: Number,
    required: true
  },
  category: {
    type: String,
    enum: ['技巧教学', '比赛集锦', '训练方法', '战术分析', '娱乐'],
    required: true
  },
  tags: [{
    type: String,
    trim: true
  }],
  analysis: {
    isAnalyzed: { type: Boolean, default: false },
    analysisData: {
      playerActions: [{
        action: String,
        timestamp: Number,
        confidence: Number,
        position: { x: Number, y: Number }
      }],
      matchStats: {
        totalPasses: Number,
        successfulPasses: Number,
        shots: Number,
        goals: Number,
        assists: Number,
        distance: Number
      },
      techniques: [{
        name: String,
        count: Number,
        successRate: Number
      }]
    },
    processingStatus: {
      type: String,
      enum: ['pending', 'processing', 'completed', 'failed'],
      default: 'pending'
    }
  },
  engagement: {
    views: { type: Number, default: 0 },
    likes: [{ type: mongoose.Schema.Types.ObjectId, ref: 'User' }],
    dislikes: [{ type: mongoose.Schema.Types.ObjectId, ref: 'User' }],
    comments: [{
      user: { type: mongoose.Schema.Types.ObjectId, ref: 'User' },
      text: { type: String, required: true, maxlength: 500 },
      createdAt: { type: Date, default: Date.now }
    }],
    shares: { type: Number, default: 0 }
  },
  visibility: {
    type: String,
    enum: ['public', 'private', 'unlisted'],
    default: 'public'
  },
  isFeatured: { type: Boolean, default: false },
  createdAt: { type: Date, default: Date.now },
  updatedAt: { type: Date, default: Date.now }
});

// 更新时间戳
videoSchema.pre('save', function(next) {
  this.updatedAt = Date.now();
  next();
});

// 获取公开信息
videoSchema.methods.toPublicJSON = function() {
  return {
    _id: this._id,
    title: this.title,
    description: this.description,
    uploadedBy: this.uploadedBy,
    videoUrl: this.videoUrl,
    thumbnailUrl: this.thumbnailUrl,
    duration: this.duration,
    category: this.category,
    tags: this.tags,
    analysis: this.analysis,
    engagement: {
      views: this.engagement.views,
      likesCount: this.engagement.likes.length,
      dislikesCount: this.engagement.dislikes.length,
      commentsCount: this.engagement.comments.length,
      shares: this.engagement.shares
    },
    visibility: this.visibility,
    isFeatured: this.isFeatured,
    createdAt: this.createdAt
  };
};

module.exports = mongoose.model('Video', videoSchema);