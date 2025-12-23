# 🚀 Quick Guide: Upload to GitHub

## Step 1: Create GitHub Repository

1. Go to https://github.com and sign in
2. Click **"+"** → **"New repository"**
3. Name: `smart-meal-coach` (or your preferred name)
4. Description: `AI-powered nutrition tracking system using computer vision`
5. Choose **Public** (for portfolio) or **Private**
6. **Don't** check "Initialize with README" (we have one)
7. Click **"Create repository"**

## Step 2: Upload Your Code

Open PowerShell/Terminal in your project folder and run:

```powershell
# Navigate to project (if not already there)
cd "C:\Users\K sanjay\Downloads\computervision"

# Initialize git repository
git init

# Add all files
git add .

# Create first commit
git commit -m "Initial commit: Smart Meal Coach - AI nutrition tracking system"

# Add your GitHub repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/smart-meal-coach.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Step 3: Verify

1. Go to your GitHub repository page
2. You should see all your files
3. README.md will display automatically

## What Gets Uploaded

✅ **Included:**
- All source code (`src/`)
- Training scripts
- Documentation
- Requirements.txt
- Configuration files

❌ **Excluded (via .gitignore):**
- Virtual environment (`venv/`)
- Python cache files
- Model files (optional)
- Large data files (optional)

## Troubleshooting

**"Repository not found" error:**
- Check your GitHub username is correct
- Make sure repository exists on GitHub
- Verify you're logged into GitHub

**"Authentication failed":**
- Use GitHub Personal Access Token instead of password
- Or use GitHub Desktop app

**"Large file" error:**
- Don't upload model files (they're in .gitignore)
- Don't upload large image datasets

## Next Steps

1. **Add topics** to your repository:
   - `deep-learning`, `computer-vision`, `pytorch`, `streamlit`, `nutrition`

2. **Add a demo** (optional):
   - Deploy to Streamlit Cloud: https://streamlit.io/cloud
   - Add demo link to README

3. **Share your project:**
   - Add to portfolio
   - Share on LinkedIn
   - Include in resume

---

**Need help?** Check `.github_setup.md` for detailed instructions.

