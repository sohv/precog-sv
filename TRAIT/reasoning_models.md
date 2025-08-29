# TRAIT Reasoning Model Evaluation

## Installation

Same requirements as base TRAIT:
```bash
pip install -r requirements.txt
```

Additional dependencies for analysis:
```bash
pip install matplotlib pandas
```

## Usage

Complete Workflow:

1. Run reasoning evaluation
```bash
  python run_reasoning_simple.py --model_name Chatgpt --model_name_short
  Chatgpt --prompt_type 1
```

2. Analyze and save to file for plotting
```bash
  python analysis_simple.py --model_name Chatgpt --prompt_type 1 >
  chatgpt_reasoning_results.txt
```
3. Create beautiful radar plots
```bash
  python plot.py --num 1 chatgpt_reasoning_results.txt
```

### For comparing multiple models:

Run evaluations for different models
```bash
  python run_reasoning_simple.py --model_name Chatgpt --model_name_short
  Chatgpt --prompt_type 1
  python run_reasoning_simple.py --model_name gpt-4o --model_name_short
  gpt4 --prompt_type 1
```

Analyze each
```bash
  python analysis_simple.py --model_name Chatgpt --prompt_type 1 >
  chatgpt_traits.txt
  python analysis_simple.py --model_name gpt4 --prompt_type 1 >
  gpt4_traits.txt
```

Compare in radar plot
```bash
  python plot.py --num 2 chatgpt_traits.txt gpt4_traits.txt
```

What the plots will show:
1. Beautiful radar charts with personality trait profiles
2. Big Five traits: Agreeableness, Conscientiousness, Extraversion, Neuroticism, Openness
3. Dark Triad traits: Psychopathy, Machiavellianism, Narcissism
4. Color-coded comparisons between different reasoning models
5. Numerical summaries showing exact scores and differences