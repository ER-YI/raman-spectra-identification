const express = require('express');
const jwt = require('jsonwebtoken');
const { body, validationResult } = require('express-validator');
const User = require('../models/User');
const auth = require('../middleware/auth');

const router = express.Router();

// 用户注册
router.post('/register', [
  body('username').isLength({ min: 3, max: 30 }).trim().escape(),
  body('email').isEmail().normalizeEmail(),
  body('password').isLength({ min: 6 })
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }

    const { username, email, password } = req.body;

    // 检查用户是否已存在
    const existingUser = await User.findOne({
      $or: [{ email }, { username }]
    });

    if (existingUser) {
      return res.status(400).json({ 
        message: '用户名或邮箱已存在' 
      });
    }

    // 创建新用户
    const user = new User({ username, email, password });
    await user.save();

    // 生成JWT token
    const token = jwt.sign(
      { userId: user._id },
      process.env.JWT_SECRET,
      { expiresIn: '7d' }
    );

    res.status(201).json({
      message: '注册成功',
      token,
      user: user.toPublicJSON()
    });
  } catch (error) {
    console.error('注册错误:', error);
    res.status(500).json({ message: '服务器错误' });
  }
});

// 用户登录
router.post('/login', [
  body('email').isEmail().normalizeEmail(),
  body('password').exists()
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }

    const { email, password } = req.body;

    // 查找用户
    const user = await User.findOne({ email });
    if (!user) {
      return res.status(401).json({ message: '邮箱或密码错误' });
    }

    // 验证密码
    const isMatch = await user.comparePassword(password);
    if (!isMatch) {
      return res.status(401).json({ message: '邮箱或密码错误' });
    }

    // 检查用户状态
    if (!user.isActive) {
      return res.status(401).json({ message: '账户已被禁用' });
    }

    // 更新最后登录时间
    user.lastLogin = Date.now();
    await user.save();

    // 生成JWT token
    const token = jwt.sign(
      { userId: user._id },
      process.env.JWT_SECRET,
      { expiresIn: '7d' }
    );

    res.json({
      message: '登录成功',
      token,
      user: user.toPublicJSON()
    });
  } catch (error) {
    console.error('登录错误:', error);
    res.status(500).json({ message: '服务器错误' });
  }
});

// 获取当前用户信息
router.get('/me', auth, async (req, res) => {
  try {
    const user = await User.findById(req.userId)
      .populate('social.followers', 'username profile.avatar')
      .populate('social.following', 'username profile.avatar');
    
    if (!user) {
      return res.status(404).json({ message: '用户不存在' });
    }

    res.json({ user: user.toPublicJSON() });
  } catch (error) {
    console.error('获取用户信息错误:', error);
    res.status(500).json({ message: '服务器错误' });
  }
});

// 更新用户资料
router.put('/profile', auth, [
  body('profile.firstName').optional().trim().escape(),
  body('profile.lastName').optional().trim().escape(),
  body('profile.bio').optional().isLength({ max: 500 }).trim(),
  body('profile.position').optional().isIn(['前锋', '中场', '后卫', '门将', '自由人']),
  body('profile.skillLevel').optional().isIn(['初级', '中级', '高级', '专业']),
  body('profile.age').optional().isInt({ min: 16, max: 100 }),
  body('profile.location').optional().trim().escape()
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }

    const user = await User.findById(req.userId);
    if (!user) {
      return res.status(404).json({ message: '用户不存在' });
    }

    // 更新资料
    Object.keys(req.body.profile || {}).forEach(key => {
      if (req.body.profile[key] !== undefined) {
        user.profile[key] = req.body.profile[key];
      }
    });

    await user.save();

    res.json({
      message: '资料更新成功',
      user: user.toPublicJSON()
    });
  } catch (error) {
    console.error('更新资料错误:', error);
    res.status(500).json({ message: '服务器错误' });
  }
});

module.exports = router;