from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline, set_seed
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("xlnet-large-cased")

model = AutoModelWithLMHead.from_pretrained("xlnet-large-cased")

generator = pipeline('text-generation', model='xlnet-large-cased')

set_seed(42)

environment = generator("My ideal work environment is", max_length=50, num_return_sequences=100)
boss = generator('My ideal boss is', max_length=50, num_return_sequences=50)
supervisor = generator('My ideal supervisor is', max_length=50, num_return_sequences=50)
feedback0 = generator('I prefer to get my feedback in formal reviews', max_length=50, num_return_sequences=50)
feedback1 = generator('I prefer to get my feedback in informal meetings', max_length=50, num_return_sequences=50)
team = generator('I prefer to work in a team', max_length=50, num_return_sequences=25)
teamwork = generator('I prefer teamwork', max_length=50, num_return_sequences=25)
alone = generator('I prefer to work alone', max_length=50, num_return_sequences=25)
myself = generator('I prefer to work by myself', max_length=50, num_return_sequences=25)
coworkers = generator('My coworkers would describe me as', max_length=50, num_return_sequences=100)
stress1 = generator('I deal with stress ', max_length=50, num_return_sequences=25)
stress2 = generator('I handle stress ', max_length=50, num_return_sequences=25)
stress3 = generator('I manage stress ', max_length=50, num_return_sequences=25)
stress4 = generator('My way of handling stress is  ', max_length=50, num_return_sequences=25)
work = generator('I want to work here because', max_length=50, num_return_sequences=100)
motivation1 = generator('What motivates me the most is', max_length=50, num_return_sequences=50)
motivation2 = generator('WMy biggest motivation is', max_length=50, num_return_sequences=50)
work_life = generator('Work-life balance for me is', max_length=50, num_return_sequences=100)

env_df = pd.DataFrame()
for i in range(len(environment)):
    env_df.append([environment[i]['generated_text']])
boss_df = pd.DataFrame()
for i in range(len(boss)):
    boss_df.append([boss[i]['generated_text']])
    boss_df.append([supervisor[i]['generated_text']])
feedback_df = pd.DataFrame()
for i in range(len(feedback0)):
    feedback_df.append([feedback0[i]['generated_text']])
    feedback_df.append([feedback1[i]['generated_text']])
team_df = pd.DataFrame()
for i in range(len(team)):
    team_df.append([team[i]['generated_text']])
    team_df.append([teamwork[i]['generated_text']])
    team_df.append([alone[i]['generated_text']])
    team_df.append([myself[i]['generated_text']])
coworkers_df = pd.DataFrame()
for i in range(len(coworkers)):
    coworkers_df.append([coworkers[i]['generated_text']])
stress_df = pd.DataFrame()
for i in range(len(stress1)):
    stress_df.append([stress1[i]['generated_text']])
    stress_df.append([stress2[i]['generated_text']])
    stress_df.append([stress3[i]['generated_text']])
    stress_df.append([stress4[i]['generated_text']])
work_df = pd.DataFrame()
for i in range(len(work)):
    work_df.append([work[i]['generated_text']])
motivation_df = pd.DataFrame()
for i in range(len(motivation1)):
    motivation_df.append([motivation1[i]['generated_text']])
    motivation_df.append([motivation2[i]['generated_text']])
work_life_df = pd.DataFrame()
for i in range(len(work_life)):
    work_life_df.append([work_life[i]['generated_text']])

env_df.to_csv('env_df.csv', index=False)
boss_df.to_csv('boss_df.csv', index=False)
feedback_df.to_csv('feedback_df.csv', index=False)
team_df.to_csv('team_df.csv', index=False)
coworkers_df.to_csv('coworkers_df.csv', index=False)
stress_df.to_csv('stress_df.csv', index=False)
work_df.to_csv('work_df.csv', index=False)
motivation_df.to_csv('motivation_df', index=False)
work_life_df.to_csv('worklife_df.csv', index=False)

