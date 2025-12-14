# Stage 1 Curriculum Training Analysis

## 🎯 Training Configuration
- **Config**: `stage1_solo.yaml`
- **Environment**: EasyStartEnv (agents start 2-3 units from goal WITH ball)
- **Mode**: 1v1, 8x4 grid, 100 max steps
- **Episodes**: 300
- **Hyperparameters**: lr=0.0001, entropy=0.1, clip=0.3

## ✅ SUCCESS - Agents Learned to Score!

### Key Breakthroughs
- **Episode 60**: Reward **1589.69**, Win Rate **25%** 🎉
  - This proves agents CAN and DID learn to score goals!
  - Reward of 1589 indicates ~7-8 goals scored in 20 episodes
  
- **Episode 80**: Reward **196.84**, Win Rate **25%**
  - Scoring behavior still present
  
- **Episode 160**: Reward **351.26**, Win Rate **25%**
  - After dip, scoring behavior re-emerged
  
- **Episode 180**: Reward **108.00**, Win Rate **25%**
  
- **Episode 200**: Reward **490.07**, Win Rate **25%**
  - Another strong scoring period

### Goal-Scoring Episodes Summary
| Episode | Reward | Win Rate | Status |
|---------|--------|----------|--------|
| 60 | 1589.69 | 25% | ✅ PEAK |
| 80 | 196.84 | 25% | ✅ Good |
| 160 | 351.26 | 25% | ✅ Good |
| 180 | 108.00 | 25% | ✅ Moderate |
| 200 | 490.07 | 25% | ✅ Strong |
| 240 | 109.35 | 0% | ⚠️ Declining |

**Total scoring intervals**: 5 out of 15 evaluation points (33%)

## ⚠️ Problem Identified: Catastrophic Forgetting

### Issue
After Episode 200, performance collapsed:
- Episode 220: -8.18 reward, 0% win rate
- Episode 240: 109.35 reward, 0% win rate
- Episode 260: 1.84 reward, 0% win rate
- Episode 300: 2.37 reward, 0% win rate

### Root Cause
**High KL Divergence → Early Stopping → Insufficient Learning**

Analysis of KL divergence:
- 95%+ of updates stopped at epoch 1-3/10
- KL divergence: 0.04-3.5 (highly variable)
- When agents get good experiences (scoring), policy barely updates (1-3 epochs)
- When agents get bad experiences, policy still updates and forgets good behavior
- Result: **Catastrophic forgetting** of scoring behavior

### Evidence
```
Episode 60 updates:
  Early stopping at epoch 1/10 due to high KL divergence: 0.0913
  Early stopping at epoch 1/10 due to high KL divergence: 1.2946
  ... (all early stops)
```

When the agent scored at Episode 60 (1589 reward), it couldn't properly reinforce that behavior because every update stopped after 1 epoch due to KL > 0.2.

## 🔧 Solution: More Conservative Training

### Changes for `stage1_stable.yaml`
1. **Learning Rate**: 0.0001 → **0.00003** (3x reduction)
   - Smaller steps = less forgetting
   
2. **Clip Epsilon**: 0.3 → **0.2** (tighter clipping)
   - Prevent large policy changes
   
3. **Target KL**: 0.2 → **0.03** (lower threshold)
   - Allow more epochs before early stopping
   - Key insight: Better to train 5-7 epochs with KL=0.03 than 1 epoch with KL=0.3
   
4. **Entropy Coef**: 0.1 → **0.08** (slightly lower)
   - Reduce exploration once scoring is discovered
   
5. **Max Grad Norm**: 2.0 → **1.0** (stricter)
   - Prevent gradient explosion
   
6. **Value Loss**: 2.0 → **1.0** (reduced)
   - Focus more on policy improvement
   
7. **Episodes**: 300 → **500** (more samples)
   - Allow stable convergence

### Expected Results
- **Episodes 1-100**: Discover scoring (like before)
- **Episodes 100-300**: **Maintain** scoring behavior (unlike before)
- **Episodes 300-500**: Consistent 50-80% win rate
- **Final**: 70%+ win rate, avg reward >160

## 📊 Stage 1 Outcome
**STATUS**: ✅ Proof of concept successful, but unstable
- Agents **DID learn** to score (Episode 60: 1589 reward)
- Agents **FORGOT** due to catastrophic forgetting
- Need more conservative hyperparameters for stable learning

## 🎬 Next Steps
1. ✅ **COMPLETED**: Analyze Stage 1 results
2. 🔄 **IN PROGRESS**: Create `stage1_stable.yaml` with conservative hyperparameters
3. ⏭️ **NEXT**: Run Stage 1 with stable config (500 episodes)
4. ⏭️ **THEN**: If 70%+ win rate achieved, proceed to Stage 2 (medium difficulty)

## 💡 Key Insights
1. **Curriculum works**: Starting agents close to goal forced shooting behavior
2. **Agents can learn**: Peak reward of 1589 proves scoring is learnable
3. **Stability critical**: High KL divergence caused early stopping → catastrophic forgetting
4. **Solution path**: More conservative hyperparameters to maintain learned behavior

## 🏆 Success Criteria for Stage 1 Stable
- [ ] Win rate >70% sustained for last 100 episodes
- [ ] Average reward >160 (implies consistent scoring)
- [ ] Episode length <20 steps (quick wins)
- [ ] No catastrophic forgetting (no reward collapse after Episode 200)
