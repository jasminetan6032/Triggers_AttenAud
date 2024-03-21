#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 13:24:41 2023

@author: jwt30
"""
import pandas
import mne
import numpy
import quick_analyse_config as cfg
import datetime
import pytz
from termcolor import colored
import scipy
import os

def find_response_triggers(events):
    events_data = pandas.DataFrame(events)
    events_data.columns = ['sample', 'initialState','trigger']
    trigger_counts = events_data['trigger'].value_counts()
    #check for largest number of triggers (leave case for those that are missing one side - eg. participant 075801)
    likely_response_triggers = trigger_counts[trigger_counts.index > 255].nlargest(2).index
    
    if len(likely_response_triggers) == 2:
        if numpy.any(likely_response_triggers[0] == cfg.right_responses):
            cfg.event_dict.update({'response/right':likely_response_triggers[0]})
            cfg.event_dict.update({'response/left':likely_response_triggers[1]})
        elif numpy.any(likely_response_triggers[0] == cfg.left_responses):
            cfg.event_dict.update({'response/right':likely_response_triggers[1]})
            cfg.event_dict.update({'response/left':likely_response_triggers[0]})

    elif len(likely_response_triggers) == 1:
        if numpy.any(likely_response_triggers[0] == cfg.right_responses):
            cfg.event_dict.update({'response/right':likely_response_triggers[0]})
            cfg.event_dict.pop('response/left')
            print("Warning: Missing left response")

        elif numpy.any(likely_response_triggers[0] == cfg.left_responses):
            cfg.event_dict.update({'response/left':likely_response_triggers[0]})
            cfg.event_dict.pop('response/right')
            print("Warning: Missing right response")
            
    elif len(likely_response_triggers) == 0:
        print("Warning: Missing all response triggers")

def fix_old_triggers(events):
    #the first version of the code used the following triggers for deviants. This code replaces them with the currently used triggers for easier analysis.
    old_triggers = [131,132,133,134,128]
    old_triggers_to_fix = [i for i in range(0, len(events)) if numpy.any(events[i,2]== old_triggers)]
    if len(old_triggers_to_fix) > 0:
        events[events[:,2] == 131,2] = 35
        events[events[:,2] == 132,2] = 36
        events[events[:,2] == 133,2] = 37
        events[events[:,2] == 134,2] = 38 
        events[events[:,2] == 128,2] = 32 
    return events

def shift_triggers(raw,events):
    #The previous experimental code sent the triggers at sound offset rather than onset. This function shifts triggers for standards, deviants and targets by 50ms to the onset of the sound.  
    date = raw.info['meas_date']
    trigger_correction_date = datetime.datetime(2023,5,17,tzinfo=pytz.utc)
    if date < trigger_correction_date:
        beeps_standards_devs = [i for i in range(0, len(events)) if (numpy.all(events[i,2]!= [25,26,27,28]) and numpy.all(events[i,2] != cfg.all_responses))] 
        for i in beeps_standards_devs:
            events[i,0] = events[i,0] - 50
        #novels = [i for i in range(0, len(events)) if numpy.any(events[i,2]== [25,26,27,28])] 

def remove_spurious_triggers(events):
    spurious_triggers = [i for i in range(0, len(events)) if events[i,1]!=0 and events[i,1]==events[i-1,2] and events[i,0] - events[i-1,0]==1]
    trigger_to_remove = [i - 1 for i in spurious_triggers]
    events = numpy.delete(events, trigger_to_remove ,axis = 0)
    for i in trigger_to_remove:
        events[i,1] = 0
    return events                     

def check_triggers(events):
    #check for number of stimuli and response triggers. If it doesn't equal to 288 for AttenAud and 168 for AttenVis (per run), or if there are an unreasonable number of response triggers, flag to take a look.
    stimuli_list = events[:, 2].tolist()
    list_of_stimuli_triggers = [i for i in stimuli_list if i < 255]
    nStimuli = len(list_of_stimuli_triggers)
    if nStimuli != cfg.number_of_stimuli:
        warning = 'Unexpected number of stimuli triggers: '
        print(colored(warning + str(len(list_of_stimuli_triggers)),'magenta'))
    
    if cfg.paradigm == 'AttenVis':
        stim_triggers = numpy.array([i for i in stimuli_list if i < 32])
        target_triggers = len(numpy.array([i for i in stimuli_list if i == 32]))
    elif cfg.paradigm in ('AttenAud','Misophonia'):
        stim_triggers = numpy.array([i for i in stimuli_list if i < 100])
        targets = [val for key, val in cfg.event_dict.items() if 'target' in key]
        target_triggers = len(numpy.array([i for i in stimuli_list if i in targets]))
    
    response_triggers = [i for i in stimuli_list if i > 255] 
    
    #check column 2 for any triggers that are not either events in dictionary or responses
    problematic_events = numpy.empty((0,3),dtype=numpy.int64)
    uncorrected_events = set(stimuli_list) - set(list(cfg.event_dict.values()) + cfg.all_responses)
    for problematic_trigger in uncorrected_events:
        problem_trigger=events[events[:,2]==problematic_trigger,:]
        problematic_events = numpy.concatenate((problematic_events,problem_trigger),axis=0)

    # #check column two for any triggers that do not start from zero and extract events to eyeball
    # non_zero_initial_states = [i for i in range(0, len(events)) if not events[i,1]==0]
    # problematic_events = events[non_zero_initial_states,:]
    return problematic_events, nStimuli, stim_triggers, target_triggers,response_triggers

        
def fix_superimposed_triggers(events):
    
    non_zero_initial_states = [i for i in range(0, len(events)) if events[i,1]!=0]
    
    for i in non_zero_initial_states:
        events[i,2] = events[i,2] - events[i,1] 
        
    return events

# def find_sticky_triggers(events,trigger):
#     #check for range of gaps between sticky triggers after deleting stimuli triggers
#     events_del = numpy.delete(events, numpy.where(events[:,1]!=0),axis = 0)
    
#     sticky_trigger = events_del[events_del[:,2] == trigger]
#     event_sample_number = sticky_trigger[:,0]

#     sample_differences = numpy.diff(event_sample_number).tolist()
#     sample_differences.insert(0, 0)

#     sample_differences = numpy.array(sample_differences)
#     sample_differences = numpy.reshape(sample_differences,(sample_differences.size,1))
#     sticky_trigger_with_difference = numpy.append(sticky_trigger,sample_differences, axis = 1)
#     stuck_responses = sticky_trigger_with_difference[sticky_trigger_with_difference[:,3] < 10000]
#     stuck_responses = stuck_responses[1:,]
#     if len(stuck_responses)>1:
#         sticky_trigger_dict = {'trigger': trigger, 
#                                'max_sample_difference': numpy.max(stuck_responses[1:,3]),
#                                'min_sample_difference': numpy.min(stuck_responses[1:,3]),
#                                'mean_sample_difference': numpy.mean(stuck_responses[1:,3]),
#                                'median_sample_difference': numpy.median(stuck_responses[1:,3])}
#     else:
#         sticky_trigger_dict = {'trigger': trigger, 
#                                'max_sample_difference': 0,
#                                'min_sample_difference': 0,
#                                'mean_sample_difference': 0,
#                                'median_sample_difference': 0}

#     sticky_trigger_array = numpy.array(list(sticky_trigger_dict.items()))
#     return sticky_trigger_array

    
# def remove_stuck_triggers(events):
#     #stuck triggers are defined as any response triggers that follow the same response trigger or a trigger corrected from an initial state of the same response trigger
#     stuck_triggers_right = [i for i in range(0, len(events)) if numpy.any(events[i,2]== cfg.event_dict['response/right']) and (numpy.any(events[i-1,2]== cfg.event_dict['response/right']) or numpy.any(events[i-1,1]== cfg.event_dict['response/right']))]
#     stuck_triggers_left = [i for i in range(0, len(events)) if numpy.any(events[i,2]== cfg.event_dict['response/left']) and (numpy.any(events[i-1,2]== cfg.event_dict['response/left']) or numpy.any(events[i-1,1]== cfg.event_dict['response/left']))]

#     stuck_triggers = stuck_triggers_left + stuck_triggers_right    
#     valid_triggers = [i for i in range(0,len(events)) if i not in stuck_triggers]
#     fixed_events = events[valid_triggers,:]
#     return fixed_events

def remove_stuck_triggers(events):
    #stuck triggers are defined as any response triggers that follow the same response trigger. This will remove the stream of repeated response triggers.
    try:
        stuck_triggers_right = [i for i in range(0, len(events)) if numpy.any(events[i,2]== cfg.event_dict['response/right']) and (numpy.any(events[i-1,2]== cfg.event_dict['response/right']))]
    except:
        print("missing right response")
    try:         
        stuck_triggers_left = [i for i in range(0, len(events)) if numpy.any(events[i,2]== cfg.event_dict['response/left']) and (numpy.any(events[i-1,2]== cfg.event_dict['response/left']))]
    except:
        print("missing left response")
    
    try:
        stuck_triggers = stuck_triggers_left + stuck_triggers_right
    except:
        stuck_triggers = []
        print("missing triggers")
        
    valid_triggers = [i for i in range(0,len(events)) if i not in stuck_triggers]
    fixed_events = events[valid_triggers,:]
    return fixed_events

def remove_extra_response_triggers(events):
    #This function is used to remove response triggers that may remain after obvious stuck ones are removed with the function remove_stuck_triggers. This includes:
    #1)Any response triggers that appear after the target, as the stream of sticking triggers can be interrupted by other stimuli
    #2) Any double presses, including when participants make an error. Their first response is what is counted so any self-correction is disregarded.
    response_buttons = [val for key, val in cfg.event_dict.items() if 'response' in key]
    
    if cfg.paradigm == 'AttenVis':
        extra_responses = [i for i in range(0, len(events)-1) if numpy.any(events[i,2]== list(cfg.event_dict.values())[9:11]) and (numpy.any(events[i-1,2]== 32))]
        double_presses = [i for i in range(0, len(events)-1) if numpy.any(events[i,2]== list(cfg.event_dict.values())[9:11]) and (numpy.any(events[i-1,2]== list(cfg.event_dict.values())[9:11]))]

    elif cfg.paradigm in ('AttenAud','Misophonia'):
        extra_responses = [i for i in range(0, len(events)-1) if numpy.any(events[i,2]== response_buttons) and (numpy.any(events[i-1,1]== response_buttons))]#placeholder number for now
        double_presses = [i for i in range(0, len(events)-1) if numpy.any(events[i,2]== response_buttons) and (numpy.any(events[i-1,2]== response_buttons))]

    unwanted_events = extra_responses + double_presses    
    fixed_events = numpy.delete(events,unwanted_events,axis=0)
    return fixed_events

def check_for_missing_responses(events):
    #only useable for AttenVis
    missing_responses = [i for i in range(0, len(events)-1) if numpy.any(events[i,2]== list(cfg.event_dict.values())[0:8]) and (numpy.all(events[i+1,2]!= list(cfg.event_dict.values())[9:11]))]
    return missing_responses

def correct_stuck_plus_trigger(events):
    right_triggers_to_fix = [i for i in range(0, len(events)) if numpy.any(events[i,2] > cfg.event_dict['response/right']) and (numpy.any(events[i,2] not in cfg.all_responses))]
    for i in right_triggers_to_fix:
        events[i,2]=events[i,2]-cfg.event_dict['response/right']

    left_triggers_to_fix = [i for i in range(0, len(events)) if numpy.any(events[i,2] > cfg.event_dict['response/left']) and (numpy.any(events[i,2] not in cfg.all_responses))]
    for i in left_triggers_to_fix:
        events[i,2]=events[i,2]-cfg.event_dict['response/left']
        return events

#load alignment csv file
df = pandas.read_csv ('/local_mount/space/hypatia/2/users/Jasmine/AttenAud/updated_meg_mri_alignment_20240306.csv')

for i in range(0,len(df)):
    file = df.loc[i,'Paradigm_data_path']    
    raw = mne.io.read_raw_fif(file)
    #check if file exists
    events_fname = file.replace('_raw.fif','_fixed_eve.fif')
    if os.path.isfile(events_fname): # and not writefile:
        print('>>> Fixed event file for subject already exists')
    else: 
        try:
            events = mne.find_events(raw, stim_channel='STI101',uint_cast=True)
            df.at[i,'errors'] = 'no input error'
        except:
            events = mne.find_events(raw, stim_channel='STI101',shortest_event=1, uint_cast=True)
            df.at[i,'errors'] = 'ValueError:shortest_event=1'
            
        find_response_triggers(events)
            
        events = remove_spurious_triggers(events)
                
        fixed_events = fix_superimposed_triggers(events)
        fixed_events = remove_stuck_triggers(fixed_events)
        fixed_events = correct_stuck_plus_trigger(fixed_events)
        fixed_events = fix_old_triggers(fixed_events)
        fixed_events = remove_extra_response_triggers(fixed_events)
        # missing_responses = check_for_missing_responses(fixed_events)
    
                
        # if missing_responses:
        #     df.at[i,'missing_responses'] = missing_responses
        # else:
        #     df.at[i,'missing_responses'] = ""
    
         
        new_problematic_events, new_nStimuli, StimTriggers, nTargetTriggers,response_triggers = check_triggers(fixed_events)
            
        df.at[i,'new_nStimuli'] = new_nStimuli
        df.at[i,'nStimTriggers'] = len(StimTriggers)
        df.at[i,'nTargetTriggers'] = nTargetTriggers
        df.at[i,'nResponseTriggers'] = len(response_triggers)
        
        mat_file   = file.replace('_raw.fif','_behaviour.mat')
        
        try:
            mat = scipy.io.loadmat(mat_file)
            if cfg.paradigm == 'AttenVis':
                mat_triggers = numpy.transpose(mat['triggers'][0])
            elif cfg.paradigm in ('AttenAud','Misophonia'):
                mat_triggers = numpy.squeeze(mat['Triggers_trace'])
                mat_responses = numpy.squeeze(mat['Response_trace'])
                response_count = pandas.Series(mat_responses[:,0]).value_counts()
                if response_count[1.0]+response_count[1000.0] == len(response_triggers):
                    print("Responses from Matlab and MNE match up.")
                    df.at[i,'responses warning'] = ''
                else:
                    print("Responses from Matlab and MNE do not match up.")
                    df.at[i,'responses warning'] = 'check responses'
                    
            #df.at[i,'mat_triggers'] = mat_triggers.astype(object)
    
            warning_triggers = numpy.array_equal(mat_triggers,StimTriggers)
            
            if warning_triggers:
                print("Stimuli triggers match triggers from Matlab code.")
                df.at[i,'warning']= ""
            else:
                if len(mat_triggers) != len(StimTriggers):
                    df.at[i,'warning']= "missing stimuli"
                else: 
                    diff = (~numpy.equal(mat_triggers, StimTriggers)).astype(int)
                    if cfg.paradigm == 'AttenVis':
                        stimuli_events = fixed_events[fixed_events[:,2]<32,:]
                    elif cfg.paradigm in ('AttenAud','Misophonia'):
                        stimuli_events = fixed_events[fixed_events[:,2]<100,:]
                    differing_events = stimuli_events[diff==1,:]
                    df.at[i,'warning']= differing_events
                    
        except:
            print("no available behavioural .mat file")
    
        
        # try:
        #     mne.write_events(events_fname,fixed_events)
        # except:
        #     print("Destination file exists.")    
    
    
    # if cfg.event_dict.get('response/left'):
    #     df.at[i,'sticky_triggers_left'] = find_sticky_triggers(events, cfg.event_dict['response/left']).astype(object)
    # if cfg.event_dict.get('response/right'):    
    #     df.at[i,'sticky_triggers_right'] = find_sticky_triggers(events, cfg.event_dict['response/right']).astype(object)
        
    
df.to_csv('/local_mount/space/hypatia/2/users/Jasmine/AttenAud/updated_meg_mri_alignment_20240306_check_triggers.csv')


