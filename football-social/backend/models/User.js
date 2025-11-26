const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');

const userSchema = new mongoose.Schema({
  username: {
    type: String,
    required: true,
    unique: true,
    trim: true,
    minlength: 3,
    maxlength: 30
  },
  email: {
    type: String,
    required: true,
    unique: true,
    lowercase: true,
    trim: true
  },
  password: {
    type: String,
    required: true,
    minlength: 6
  },
  profile: {
    firstName: { type: String, trim: true },
    lastName: { type: String, trim: true },
    avatar: { type: String, default: '' },
    bio: { type: String, maxlength: 500 },
    position: { 
      type: String, 
      enum: ['前锋', '中场', '后卫', '门将', '自由人'],
      default: '自由人'
    },
    skillLevel: { 
      type: String, 
      enum: ['初级', '中级', '高级', '专业'],
      default: '中级'
    },
    age: { type: Number, min: 16, max: 100 },
    location: { type: String, trim: true }
  },
  stats: {
    matchesPlayed: { type: Number, default: 0 },
    goals: { type: Number, default: 0 },
    assists: { type: Number, default: 0 },
    wins: { type: Number, default: 0 },
    losses: { type: Number, default: 0 },
    rating: { type: Number, default: 0, min: 0, max: 10 }
  },
  social: {
    followers: [{ type: mongoose.Schema.Types.ObjectId, ref: 'User' }],
    following: [{ type: mongoose.Schema.Types.ObjectId, ref: 'User' }],
    favoriteVideos: [{ type: mongoose.Schema.Types.ObjectId, ref: 'Video' }]
  },
  isVerified: { type: Boolean, default: false },
  isActive: { type: Boolean, default: true },
  lastLogin: { type: Date },
  createdAt: { type: Date, default: Date.now },
  updatedAt: { type: Date, default: Date.now }
});

// 密码加密中间件
userSchema.pre('save', async function(next) {
  if (!this.isModified('password')) return next();
  
  try {
    const salt = await bcrypt.genSalt(10);
    this.password = await bcrypt.hash(this.password, salt);
    next();
  } catch (error) {
    next(error);
  }
});

// 密码验证方法
userSchema.methods.comparePassword = async function(candidatePassword) {
  return await bcrypt.compare(candidatePassword, this.password);
};

// 更新时间戳
userSchema.pre('save', function(next) {
  this.updatedAt = Date.now();
  next();
});

// 获取公开信息
userSchema.methods.toPublicJSON = function() {
  return {
    _id: this._id,
    username: this.username,
    profile: this.profile,
    stats: this.stats,
    social: {
      followersCount: this.social.followers.length,
      followingCount: this.social.following.length
    },
    isVerified: this.isVerified,
    createdAt: this.createdAt
  };
};

module.exports = mongoose.model('User', userSchema);