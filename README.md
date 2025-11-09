# Streamlit Cloud 部署指南

## 📋 目录

- [简介](#简介)
- [部署前准备](#部署前准备)
- [部署步骤](#部署步骤)
- [常见问题](#常见问题)
- [文件说明](#文件说明)

---

## 📖 简介

本部署包专为 **Streamlit Cloud** 部署设计。Streamlit Cloud 是 Streamlit 官方提供的免费云托管服务，可以轻松部署 Streamlit 应用，无需配置服务器。

### 为什么选择 Streamlit Cloud？

- ✅ **完全免费**：个人项目永久免费
- ✅ **零配置**：无需配置服务器、域名、SSL证书
- ✅ **自动部署**：连接到 GitHub 仓库后自动部署
- ✅ **自动更新**：代码推送后自动更新
- ✅ **HTTPS支持**：自动提供 HTTPS 访问
- ✅ **全球CDN**：快速访问速度

---

## 🚀 部署前准备

### 1. 准备 GitHub 账号

如果没有 GitHub 账号，请先注册：
- 访问：https://github.com
- 点击 "Sign up" 注册账号

### 2. 安装 Git

如果还没有安装 Git，请下载安装：
- Windows: https://git-scm.com/download/win
- Mac: `brew install git`
- Linux: `sudo apt-get install git`

### 3. 准备项目文件

确保 `streamlit_cloud_deploy` 文件夹中包含以下文件：

```
streamlit_cloud_deploy/
├── app.py                    # 主应用文件
├── data_processor.py         # 数据处理模块
├── metrics.py                # 评估指标模块
├── train_xgboost.py          # XGBoost训练脚本
├── train_lstm.py             # LSTM训练脚本
├── train_transformer.py      # Transformer训练脚本
├── requirements.txt          # 依赖包列表
├── 去噪后数据.xlsx          # 数据文件
├── models/                   # 模型文件夹
│   ├── xgboost_model.pkl
│   ├── lstm_model.pkl
│   └── transformer_model.pkl
├── .streamlit/               # Streamlit配置
│   └── config.toml
└── README.md                 # 本文件
```

---

## 📝 部署步骤

### 步骤 1：创建 GitHub 仓库

1. **登录 GitHub**
   - 访问 https://github.com
   - 登录您的账号

2. **创建新仓库**
   - 点击右上角的 "+" 号
   - 选择 "New repository"
   - 填写仓库信息：
     - **Repository name**: `streamlit-ml-app` (或其他名称)
     - **Description**: `机器学习模型可视化系统`
     - **Visibility**: 选择 `Public` (Streamlit Cloud 免费版需要公开仓库)
     - **不要**勾选 "Initialize this repository with a README"
   - 点击 "Create repository"

### 步骤 2：上传代码到 GitHub

#### 方法一：使用 Git 命令行（推荐）

1. **打开命令行/终端**
   - Windows: 打开 PowerShell 或 CMD
   - Mac/Linux: 打开 Terminal

2. **进入项目目录**
   ```bash
   cd "d:\Python Program\bisai\streamlit_cloud_deploy"
   ```

3. **初始化 Git 仓库**
   ```bash
   git init
   ```

4. **添加所有文件**
   ```bash
   git add .
   ```

5. **提交文件**
   ```bash
   git commit -m "Initial commit: Streamlit ML App"
   ```

6. **连接远程仓库**
   ```bash
   git remote add origin https://github.com/你的用户名/streamlit-ml-app.git
   ```
   > 注意：将 `你的用户名` 替换为您的 GitHub 用户名

7. **推送代码**
   ```bash
   git branch -M main
   git push -u origin main
   ```

8. **输入 GitHub 用户名和密码**
   - 如果提示输入用户名和密码，请输入您的 GitHub 账号信息
   - 如果使用 Personal Access Token，请使用 token 作为密码

#### 方法二：使用 GitHub Desktop（图形界面）

1. **下载 GitHub Desktop**
   - 访问：https://desktop.github.com
   - 下载并安装

2. **登录 GitHub**
   - 打开 GitHub Desktop
   - 登录您的 GitHub 账号

3. **添加仓库**
   - 点击 "File" -> "Add local repository"
   - 选择 `streamlit_cloud_deploy` 文件夹
   - 点击 "Add repository"

4. **发布仓库**
   - 点击 "Publish repository"
   - 输入仓库名称
   - 选择 "Keep this code private" 或取消选择（公开仓库）
   - 点击 "Publish repository"

#### 方法三：使用网页上传（简单但不推荐）

1. **在 GitHub 仓库页面**
   - 点击 "uploading an existing file"

2. **拖拽文件**
   - 将 `streamlit_cloud_deploy` 文件夹中的所有文件拖拽到网页

3. **提交**
   - 输入提交信息
   - 点击 "Commit changes"

### 步骤 3：部署到 Streamlit Cloud

1. **访问 Streamlit Cloud**
   - 访问：https://streamlit.io/cloud
   - 点击 "Sign up" 或 "Get started"

2. **使用 GitHub 登录**
   - 点击 "Continue with GitHub"
   - 授权 Streamlit Cloud 访问您的 GitHub 账号

3. **新建应用**
   - 点击 "New app"
   - 填写应用信息：
     - **Repository**: 选择您刚创建的仓库
     - **Branch**: 选择 `main` 或 `master`
     - **Main file path**: 输入 `app.py`
     - **App URL**: 可以自定义（可选）
   - 点击 "Deploy!"

4. **等待部署**
   - Streamlit Cloud 会自动安装依赖
   - 部署过程可能需要 2-5 分钟
   - 部署完成后，您会收到一个 URL，例如：`https://your-app-name.streamlit.app`

5. **访问应用**
   - 点击 URL 访问您的应用
   - 应用已经可以正常使用了！

### 步骤 4：验证部署

1. **检查应用是否正常运行**
   - 访问 Streamlit Cloud 提供的 URL
   - 检查页面是否正常加载
   - 测试各个功能是否正常

2. **检查日志**
   - 在 Streamlit Cloud 控制台中查看日志
   - 如果有错误，检查错误信息并修复

---

## 🔧 常见问题

### Q1: 部署失败，提示 "ModuleNotFoundError"

**原因**: 缺少依赖包

**解决方法**:
1. 检查 `requirements.txt` 文件是否包含所有依赖
2. 确保所有依赖包名称正确
3. 重新部署应用

### Q2: 模型文件找不到

**原因**: 模型文件未上传或路径不正确

**解决方法**:
1. 确保 `models/` 文件夹中包含所有模型文件
2. 检查 `app.py` 中的模型路径是否正确
3. 确保模型文件已提交到 GitHub 仓库

### Q3: 数据文件加载失败

**原因**: 数据文件未上传或路径不正确

**解决方法**:
1. 确保 `去噪后数据.xlsx` 文件已上传
2. 检查 `app.py` 中的数据文件路径是否正确
3. 确保数据文件已提交到 GitHub 仓库

### Q4: 应用运行缓慢

**原因**: 模型文件过大或计算资源不足

**解决方法**:
1. 优化模型文件大小
2. 考虑使用模型压缩技术
3. 优化代码性能

### Q5: 如何更新应用？

**方法**:
1. 修改本地代码
2. 提交并推送到 GitHub
3. Streamlit Cloud 会自动检测更改并重新部署

```bash
git add .
git commit -m "Update app"
git push
```

### Q6: 如何查看应用日志？

**方法**:
1. 在 Streamlit Cloud 控制台中
2. 点击应用名称
3. 查看 "Logs" 标签页

### Q7: 如何设置环境变量？

**方法**:
1. 在 Streamlit Cloud 控制台中
2. 点击应用名称
3. 进入 "Settings" -> "Secrets"
4. 添加环境变量

### Q8: 如何绑定自定义域名？

**注意**: Streamlit Cloud 免费版不支持自定义域名

**解决方法**:
1. 升级到 Streamlit Cloud Team 版本
2. 或使用其他部署方案（如 Heroku、Railway 等）

---

## 📁 文件说明

### 核心文件

- **app.py**: 主应用文件，包含 Streamlit 界面和业务逻辑
- **data_processor.py**: 数据处理模块，负责数据加载和预处理
- **metrics.py**: 评估指标模块，计算模型性能指标

### 训练脚本

- **train_xgboost.py**: XGBoost 模型训练脚本
- **train_lstm.py**: LSTM 模型训练脚本
- **train_transformer.py**: Transformer 模型训练脚本

### 配置文件

- **requirements.txt**: Python 依赖包列表
- **.streamlit/config.toml**: Streamlit 配置文件
- **.gitignore**: Git 忽略文件列表

### 数据文件

- **去噪后数据.xlsx**: 训练和测试数据

### 模型文件

- **models/xgboost_model.pkl**: XGBoost 模型
- **models/lstm_model.pkl**: LSTM 模型
- **models/transformer_model.pkl**: Transformer 模型

---

## 🎯 下一步

部署成功后，您可以：

1. **分享应用**: 将 Streamlit Cloud URL 分享给其他人
2. **持续更新**: 通过 Git 推送更新代码
3. **监控使用**: 在 Streamlit Cloud 控制台查看应用使用情况
4. **优化性能**: 根据使用情况优化代码和模型

---

## 📞 获取帮助

如果遇到问题，可以：

1. **查看 Streamlit Cloud 文档**: https://docs.streamlit.io/streamlit-community-cloud
2. **查看 Streamlit 文档**: https://docs.streamlit.io
3. **在 GitHub 上提交问题**: 在项目仓库中提交 Issue
4. **联系支持**: 通过 Streamlit Cloud 控制台联系支持

---

## ✅ 部署检查清单

部署前请确认：

- [ ] GitHub 账号已注册
- [ ] Git 已安装
- [ ] 所有文件已准备好
- [ ] `requirements.txt` 文件完整
- [ ] `.streamlit/config.toml` 文件存在
- [ ] 模型文件已包含
- [ ] 数据文件已包含
- [ ] 代码已推送到 GitHub
- [ ] Streamlit Cloud 账号已注册
- [ ] 应用已部署
- [ ] 应用可以正常访问

---

**祝部署顺利！** 🚀

如有任何问题，请随时查看文档或联系支持。

