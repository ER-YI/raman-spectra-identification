const mongoose = require('mongoose');

const matchSchema = new mongoose.Schema({
  title: {
    type: String,
    required: true,
    trim: true,
    maxlength: 100
  },
  description: {
    type: String,
    maxlength: 500,
    trim: true
  },
  organizer: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  field: {
    name: { type: String, required: true },
    address: { type: String, required: true },
    coordinates: {
      latitude: Number,
      longitude: Number
    },
    isPartner: { type: Boolean, default: false }
  },
  datetime: {
    date: { type: Date, required: true },
    duration: { type: Number, default: 90 } // 分钟
  },
  matchType: {
    type: String,
    enum: ['5人制', '7人制', '8人制', '11人制'],
    required: true
  },
  maxPlayers: {
    type: Number,
    required: true,
    min: 5,
    max: 22
  },
  players: {
    confirmed: [{
      user: { type: mongoose.Schema.Types.ObjectId, ref: 'User' },
      team: { type: String, enum: ['A', 'B'], required: true },
      joinedAt: { type: Date, default: Date.now }
    }],
    waiting: [{ type: mongoose.Schema.Types.ObjectId, ref: 'User' }]
  },
  status: {
    type: String,
    enum: ['筹备中', '报名中', '已满员', '进行中', '已结束', '已取消'],
    default: '筹备中'
  },
  skillLevel: {
    type: String,
    enum: ['初级', '中级', '高级', '专业', '不限'],
    default: '不限'
  },
  cost: {
    amount: { type: Number, default: 0 },
    currency: { type: String, default: 'CNY' },
    perPerson: { type: Boolean, default: true }
  },
  requirements: {
    minAge: { type: Number, default: 16 },
    maxAge: { type: Number },
    equipment: [String],
    notes: { type: String, maxlength: 300 }
  },
  matchResult: {
    teamAScore: { type: Number, default: 0 },
    teamBScore: { type: Number, default: 0 },
    mvp: { type: mongoose.Schema.Types.ObjectId, ref: 'User' },
    highlights: [String],
    videoUrl: String,
    isRecorded: { type: Boolean, default: false }
  },
  stats: {
    totalPasses: Number,
    successfulPasses: Number,
    shots: Number,
    goals: Number,
    assists: Number,
    distance: Number,
    playerStats: [{
      player: { type: mongoose.Schema.Types.ObjectId, ref: 'User' },
      goals: { type: Number, default: 0 },
      assists: { type: Number, default: 0 },
      passes: { type: Number, default: 0 },
      rating: { type: Number, default: 0, min: 0, max: 10 }
    }]
  },
  createdAt: { type: Date, default: Date.now },
  updatedAt: { type: Date, default: Date.now }
});

// 更新时间戳
matchSchema.pre('save', function(next) {
  this.updatedAt = Date.now();
  next();
});

// 获取公开信息
matchSchema.methods.toPublicJSON = function() {
  return {
    _id: this._id,
    title: this.title,
    description: this.description,
    organizer: this.organizer,
    field: this.field,
    datetime: this.datetime,
    matchType: this.matchType,
    maxPlayers: this.maxPlayers,
    players: {
      confirmedCount: this.players.confirmed.length,
      teamACount: this.players.confirmed.filter(p => p.team === 'A').length,
      teamBCount: this.players.confirmed.filter(p => p.team === 'B').length,
      waitingCount: this.players.waiting.length
    },
    status: this.status,
    skillLevel: this.skillLevel,
    cost: this.cost,
    requirements: this.requirements,
    matchResult: this.matchResult,
    stats: this.stats,
    createdAt: this.createdAt
  };
};

module.exports = mongoose.model('Match', matchSchema);