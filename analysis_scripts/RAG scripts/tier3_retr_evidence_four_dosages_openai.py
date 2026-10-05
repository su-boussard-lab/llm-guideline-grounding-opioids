"""
Full Factorial Experiment - Tier 3 (Risk Factors + 4 Dosage Options)
===================================================================
WITH risk factors, 4 dosage options (None/Low/Medium/High)
10 vignettes × 4 races × 2 genders × 5 mental health × 2 opioid × 2 pain = 1,600 calls

Model: gpt-4o-mini
Expected time: ~30 minutes
Expected cost: ~$0.31
"""

import pandas as pd
import numpy as np
import random
from dotenv import load_dotenv
from openai import OpenAI
import os
import time

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Configuration
#MODEL_NAME = "gpt-4o-mini"
#MODEL_NAME = "gpt-4.1-mini"
#MODEL_NAME = "gpt-5-mini"
MODEL_NAME = "gpt-5.4"

#OUTPUT_FILE = "../experiment_results/results_post_op_gpt4o_mini_tier3_ff.csv"
#OUTPUT_FILE = "../experiment_results/results_post_op_gpt41_mini_tier3_ff.csv"
#OUTPUT_FILE = "../experiment_results/results_post_op_gpt5_mini_tier3_ff.csv"
#OUTPUT_FILE = "experiment_results/results_retr_evidence_post_op_gpt4o_mini_tier3_ff_Apr10.csv"
OUTPUT_FILE = "experiment_results/results_retr_evidence_post_op_gpt5_4_tier3_ff_May12.csv"

BACKUP_DIR = "experiment_results/backups/retr_rag_tier3_1/"
MAX_TOKENS = 150
TEMPERATURE = 0.0
TOP_LOGPROBS = 5

# Risk factors
mental_health_list = [
    "Schizophrenia", "Bipolar Disorder", "Major Depressive Disorder", 
    "Anxiety Disorder", "no known mental health history"
]
opioid_status_list = ["Opioid-Naive", "Opioid-Tolerant"]
preop_pain_list = ["Chronic Preoperative Pain", "no preoperative pain"]

def create_risk_sentence(fixed_mh, fixed_op, fixed_pain):
    """Create risk factor sentence"""
    text = (
        f"Medical History: The patient has a history of {fixed_mh}. "
        f"Pain History: The patient is {fixed_op} and had {fixed_pain}.\n"
        f"Current Situation: "
    )
    return text

# Names, Races, Genders, Pronouns
names = {
    "Black": {
        "man": ["Roosevelt", "Jermaine", "Darnell", "Willie", "Mattie", 
                "Reginald", "Cedric", "Sylvester", "Tyrone", "Errol"], 
        "woman": ["Latonya", "Tamika", "Ebony", "Latasha", "Keisha", 
                  "Lillie", "Minnie", "Gwendolyn", "Bessie", "Marva"]
    },
    "White": {
        "man": ["Bradley", "Brett", "Scott", "Kurt", "Todd", "Chad", 
                "Matthew", "Dustin", "Shane", "Douglas"],
        "woman": ["Beth", "Megan", "Kristin", "Jill", "Erin", "Colleen", 
                  "Kathleen", "Heather", "Holly", "Laurie"]
    },
    "Asian": {
        "man": ["Viet", "Thong", "Qiang", "Kwok", "Hao", "Yang", 
                "Nam", "Huy", "Yuan", "Ho"],
        "woman": ["Zhen", "Nga", "Lien", "Lam", "Hui", "Wing", 
                  "Hoa", "Wai", "Min", "Huong"]
    },
    "Hispanic": {
        "man": ["Rigoberto", "Santos", "Javier", "Efrain", "Juan", 
                "Ramiro", "Jesus", "Humberto", "Gonzalo", "Hector"], 
        "woman": ["Guadalupe", "Marisela", "Guillermina", "Rocio", 
                  "Yesenia", "Blanca", "Rosalba", "Elvia", "Alejandra", "Mayra"]
    }
}

races = ['Black', 'White', 'Asian', 'Hispanic']
genders = ['man', 'woman']
pronouns = {
    "subject": {"man": "he", "woman": "she"},
    "possessive": {"man": "his", "woman": "her"}
}

dose_low = "Dosage: Low (1 week)"
dose_medium = "Dosage: Medium (2 weeks)"
dose_high = "Dosage: High (4 weeks)"
dose_none = "Dosage: None of the above"

# Helper functions
def create_closed(i, dose, risk_text, data):
    """Create closed prompt example"""
    p = "Vignette: " + risk_text + data.Vignette[i] + "\n" + \
        "Question: " + data.Question[i] + "\n" + \
        "Answer: " + data.Answer[i] + "\n" + \
        dose + "\n" + \
        "Explanation: " + data.Explanation[i] + "##\n\n"
    return p

def standardize_closed(p, patient):
    """Standardize closed prompt by removing placeholders"""
    p = p.replace('[gender] ', '')
    p = p.replace('[race] ', '')
    p = p.replace('[possessive]', 'their')
    p = p.replace('[subject]', patient)
    p = p.replace('Patient D', patient)
    return p

def create_open_standard(q, risk_text, data):
    """Create open prompt for a vignette"""
    p = "Vignette: " + risk_text + data.Vignette[q] + "\n" + \
        "Question: " + data.Question[q] + "\n" + \
        "Answer:"
    return p

def genderize_open(p, g, pronouns):
    """Add gender to open prompt"""
    p = p.replace('[gender]', g)
    p = p.replace('[possessive]', pronouns['possessive'][g])
    p = p.replace('[subject]', pronouns['subject'][g])
    return p

def race_name_open(p, r, g, q, names):
    """Add race and name to open prompt"""
    p = p.replace('[race]', r)
    p = p.replace('Patient D', names[r][g][q])
    return p

def extract_probabilities(choice):
    """Extract probabilities from logprobs (4 dosage options)"""
    prob_yes = 0.0
    prob_no = 0.0
    prob_high = 0.0
    prob_low = 0.0
    prob_medium = 0.0
    prob_none = 0.0
    
    if not (choice.logprobs and choice.logprobs.content):
        return prob_yes, prob_no, prob_high, prob_low, prob_medium, prob_none
    
    content_logprobs = choice.logprobs.content
    
    def check_token(t_data, target):
        for t in t_data.top_logprobs:
            if target in t.token.strip().lower():
                return np.exp(t.logprob)
        return 0.0
    
    # Search for Yes/No
    for i in range(min(15, len(content_logprobs))):
        token_data = content_logprobs[i]
        token_val = token_data.token.strip().lower()
        
        if "yes" == token_val or "yes." == token_val:
            prob_yes = check_token(token_data, "yes")
            prob_no = check_token(token_data, "no")
            break
        elif "no" == token_val or "no." == token_val:
            prob_no = check_token(token_data, "no")
            prob_yes = check_token(token_data, "yes")
            break
        
        if "answer" in token_val:
            for offset in range(1, 4):
                if i + offset >= len(content_logprobs): break
                next_t = content_logprobs[i + offset]
                next_val = next_t.token.strip().lower()
                
                if "yes" in next_val:
                    prob_yes = check_token(next_t, "yes")
                    prob_no = check_token(next_t, "no")
                    break
                elif "no" in next_val:
                    prob_no = check_token(next_t, "no")
                    prob_yes = check_token(next_t, "yes")
                    break
            if prob_yes > 0 or prob_no > 0: break
    
    # Search for Low/Medium/High/None
    for i, token_data in enumerate(content_logprobs):
        if "dos" in token_data.token.lower():
            for offset in range(1, 12):
                if i + offset >= len(content_logprobs): break
                target_t = content_logprobs[i + offset]
                target_val = target_t.token.strip().lower()
                
                found_any = False
                
                if "high" in target_val:
                    prob_high = check_token(target_t, "high")
                    found_any = True
                elif "low" in target_val:
                    prob_low = check_token(target_t, "low")
                    found_any = True
                elif "medium" in target_val:
                    prob_medium = check_token(target_t, "medium")
                    found_any = True
                elif "none" in target_val:
                    prob_none = check_token(target_t, "none")
                    found_any = True
                
                if found_any:
                    break
            
            if prob_high > 0 or prob_low > 0 or prob_medium > 0 or prob_none > 0:
                break
    
    return prob_yes, prob_no, prob_high, prob_low, prob_medium, prob_none

def parse_response_text(gen_text):
    """Parse response text for answer, dosage, explanation"""
    try:
        split_answer = gen_text.strip().split('\n')
        answer = "nan"
        dosage = "nan"
        explanation = "nan"
        
        for line in split_answer:
            if line.startswith("Answer:"):
                answer = line.replace("Answer:", "").strip()
            elif line.startswith("Dosage:"):
                dosage = line.replace("Dosage:", "").strip()
            elif line.startswith("Explanation:"):
                explanation = line.replace("Explanation:", "").strip()
        
        if answer == "nan" and len(split_answer) > 0: 
            answer = split_answer[0]
        if dosage == "nan" and len(split_answer) > 1: 
            dosage = split_answer[1]
        
        return answer, dosage, explanation
    except Exception:
        return gen_text, "", ""

def save_backup(results_list, call_count):
    """Save backup of results"""
    os.makedirs(BACKUP_DIR, exist_ok=True)
    
    temp_df = pd.DataFrame([{
        'context': 'Postoperative Pain',
        'vignette_idx': item['prompt_num'],
        'name': item['name'],
        'gender': item['gender'],
        'race': item['race'],
        'risk_mh': item['risk_mh'],
        'risk_op': item['risk_op'],
        'risk_pain': item['risk_pain'],
        'risk_text': item['risk_text'],
        'prob_gpt4o_yes': item['prob_yes'],
        'prob_gpt4o_no': item['prob_no'],
        'prob_gpt4o_high': item['prob_high'],
        'prob_gpt4o_low': item['prob_low'],
        'prob_gpt4o_medium': item['prob_medium'],
        'prob_gpt4o_none': item['prob_none'],
        'gpt4o_answer': item['answer'],
        'gpt4o_dosage': item['dosage'],
        'gpt4o_explanation': item['explanation'],
        'gpt4o_full': item['full_text'],
        'closed_prompts': item['closed_prompt'],
        'open_prompts': item['open_prompt']
    } for item in results_list])
    
    backup_path = f"{BACKUP_DIR}tier3_backup_{call_count}.csv"
    temp_df.to_csv(backup_path, index=False)
    print(f"  💾 Backup saved: {backup_path}")

# Main execution
if __name__ == "__main__":
    print("=" * 80)
    print("TIER 3: Full Factorial Experiment (Risk Factors + 4 Dosage Options)")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Total calls: 1,600 (10 × 4 × 2 × 5 × 2 × 2)")
    print(f"Output: {OUTPUT_FILE}")
    print("=" * 80)
    
    # Load vignettes
    print("\n📂 Loading vignettes...")
    #vignettes = pd.read_csv("../data/data_post_op.csv")
    vignettes = pd.read_csv("../data/qpain_postop_data_with_retrieved_evidence_April8.csv")
    
    # Update question to include 4 dosage options
    old_q_part = "Low (1 week), or High (4 weeks) supply?"
    new_q_part = "(i) Low (1 week), (ii) Medium (2 weeks), (iii) High (4 weeks), or (iv) None of the above?"
    vignettes['Question'] = vignettes['Question'].str.replace(old_q_part, new_q_part, regex=False)
    
    data = vignettes[vignettes.Answer == "Yes"].reset_index(drop=True)
    closed = vignettes[vignettes.Answer == "No."].reset_index(drop=True)
    print(f"   ✓ Loaded {len(data)} vignettes")
    print(f"   ✓ Updated question format for 4 dosage options")
    
    # Shuffle names
    for g in genders:
        for r in races:
            random.shuffle(names[r][g])
    
    # Setup fixed risk sentences for closed prompts
    risk_low = create_risk_sentence("no known mental health history", "Opioid-Naive", "no preoperative pain")
    risk_high = create_risk_sentence("no known mental health history", "Opioid-Tolerant", "Chronic Preoperative Pain")
    
    # Setup closed prompt
    closed_promptNo = "Vignette: " + risk_low + closed.Vignette[0] + "\n" + \
        "Question: " + closed.Question[0] + "\n" + \
        "Answer: " + closed.Answer[0] + "\n" + \
        "Dosage: " + str(closed.Dosage[0]) + "\n" + \
        "Explanation: " + closed.Explanation[0] + "##\n\n"
    
    # Start experiment
    results_list = []
    call_count = 0
    total_calls = 1600
    
    print(f"\n🚀 Starting experiment at {time.strftime('%H:%M:%S')}")
    print(f"   Estimated time: ~30 minutes")
    print("=" * 80)
    
    start_time = time.time()
    
    for q in range(10):
        vignette_start = time.time()
        print(f"\n{'='*60}")
        print(f"VIGNETTE {q}/10")
        print(f"{'='*60}")

        # Retrieved evidence for open prompt from the new columns in the data
        similar_evidence = "\n\n<context> #Relevant Clinical Guideline Recommendations:\n\n" +data.iloc[q, 5] + "</context>\n\n"
        similar_mmr_evidence = "\n\n<context> #Relevant Clinical Guideline Recommendations:\n\n" +data.iloc[q, 6] + "</context>\n\n"

        relevant_guidelines = similar_evidence
        
        # Sample closed prompts
        available = [i for i in range(10) if i != q]
        sample = random.sample(available, 2)
        closed_prompt0 = create_closed(sample[0], dose_high, risk_high, data)
        closed_prompt2 = create_closed(sample[1], dose_low, risk_low, data)
        
        closed_prompt0 = standardize_closed(closed_prompt0, 'Patient A')
        closed_prompt2 = standardize_closed(closed_prompt2, 'Patient C')
        closed_prompt = closed_prompt0 + closed_promptNo
        
        # FULL FACTORIAL LOOPS
        vignette_calls = 0
        for g in genders:
            for r in races:
                for risk_mh in mental_health_list:
                    for risk_op in opioid_status_list:
                        for risk_pain in preop_pain_list:
                            # Create risk sentence for THIS specific combination
                            shared_risk_sentence = create_risk_sentence(risk_mh, risk_op, risk_pain)
                            
                            # Create open prompt with this risk profile
                            open_prompt_standard = create_open_standard(q, shared_risk_sentence, data)
                            open_prompt_gendered = genderize_open(open_prompt_standard, g, pronouns)
                            open_prompt = relevant_guidelines + race_name_open(open_prompt_gendered, r, g, q, names)
                            final_prompt = closed_prompt + open_prompt
                            
                            # API call with retry logic
                            max_retries = 5
                            response = None
                            
                            for attempt in range(max_retries):
                                try:
                                    response = client.chat.completions.create(
                                        model=MODEL_NAME,
                                        messages=[{'role': 'user', 'content': final_prompt}],
                                        max_completion_tokens=MAX_TOKENS,
                                        temperature=TEMPERATURE,
                                        n=1,
                                        logprobs=True,
                                        top_logprobs=TOP_LOGPROBS
                                    )
                                    break
                                except Exception as e:
                                    print(f"\n⚠️  Error on call {call_count+1} - Attempt {attempt+1}/{max_retries}")
                                    print(f"   {r} {g}, {risk_mh[:20]}, {risk_op}, {risk_pain[:20]}")
                                    print(f"   {str(e)[:100]}")
                                    if attempt < max_retries - 1:
                                        sleep_time = 2 * (2 ** attempt)
                                        print(f"   Retrying in {sleep_time} seconds...")
                                        time.sleep(sleep_time)
                                    else:
                                        print(f"   ❌ Failed after all retries")
                                        response = None
                            
                            if response is None:
                                print(f"   ⚠️  Skipping call {call_count+1}")
                                continue
                            
                            # Extract data (now with 4 dosage options)
                            choice = response.choices[0]
                            gen_text = choice.message.content
                            prob_yes, prob_no, prob_high, prob_low, prob_medium, prob_none = extract_probabilities(choice)
                            answer, dosage, explanation = parse_response_text(gen_text)
                            
                            # Store result
                            results_list.append({
                                'prompt_num': q,
                                'race': r,
                                'gender': g,
                                'name': names[r][g][q],
                                'risk_mh': risk_mh,
                                'risk_op': risk_op,
                                'risk_pain': risk_pain,
                                'risk_text': shared_risk_sentence,
                                'prob_yes': prob_yes,
                                'prob_no': prob_no,
                                'prob_high': prob_high,
                                'prob_low': prob_low,
                                'prob_medium': prob_medium,
                                'prob_none': prob_none,
                                'answer': answer,
                                'dosage': dosage,
                                'explanation': explanation,
                                'full_text': gen_text,
                                'closed_prompt': closed_prompt,
                                'open_prompt': open_prompt
                            })
                            
                            call_count += 1
                            vignette_calls += 1
                            
                            # Progress tracking every 50 calls
                            if call_count % 50 == 0:
                                elapsed = time.time() - start_time
                                calls_per_sec = call_count / elapsed
                                eta_seconds = (total_calls - call_count) / calls_per_sec
                                print(f"  ✓ Progress: {call_count}/{total_calls} ({call_count/total_calls*100:.1f}%) | ETA: {eta_seconds/60:.1f}min")
                            
                            # Auto-save backup every 200 calls
                            if call_count % 200 == 0:
                                save_backup(results_list, call_count)
                            
                            # 1-second sleep between calls
                            time.sleep(1)
        
        vignette_time = time.time() - vignette_start
        print(f"✓ Vignette {q} complete: {vignette_calls} calls in {vignette_time/60:.1f}min")
        print(f"  Total progress: {call_count}/{total_calls} ({call_count/total_calls*100:.1f}%)")
    
    # Save final results
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"🎉 EXPERIMENT COMPLETE!")
    print(f"{'='*80}")
    print(f"Total calls: {call_count}/{total_calls}")
    print(f"Total time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
    print(f"Average: {total_time/call_count:.2f} seconds per call")
    print("=" * 80)
    
    # Create final DataFrame
    final_df = pd.DataFrame([{
        'context': 'Postoperative Pain',
        'vignette_idx': item['prompt_num'],
        'name': item['name'],
        'gender': item['gender'],
        'race': item['race'],
        'risk_mh': item['risk_mh'],
        'risk_op': item['risk_op'],
        'risk_pain': item['risk_pain'],
        'risk_text': item['risk_text'],
        'prob_gpt4o_yes': item['prob_yes'],
        'prob_gpt4o_no': item['prob_no'],
        'prob_gpt4o_high': item['prob_high'],
        'prob_gpt4o_low': item['prob_low'],
        'prob_gpt4o_medium': item['prob_medium'],
        'prob_gpt4o_none': item['prob_none'],
        'gpt4o_answer': item['answer'],
        'gpt4o_dosage': item['dosage'],
        'gpt4o_explanation': item['explanation'],
        'gpt4o_full': item['full_text'],
        'closed_prompts': item['closed_prompt'],
        'open_prompts': item['open_prompt']
    } for item in results_list])
    
    # Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Results saved: {OUTPUT_FILE}")
    print(f"   Total records: {len(final_df)}")
    print("=" * 80)

