#!/bin/bash

# Check if the argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <number (1-11)>"
    exit 1
fi

# Store the argument
arg=$1

# Check if the argument is a valid number between 1 and 11
if [[ "$arg" =~ ^[1-9]$ ]] || [[ "$arg" == "10" ]] || [[ "$arg" == "11" ]]; then
    # Map the argument to the corresponding Python file and redirect output to text file
    case $arg in
        1)
            python 01_load_data.py > 01_load_data_OUTPUT.txt
            ;;
        2)
            python 02_explore_data.py > 02_explore_data_OUTPUT.txt
            ;;
        3)
            python 03_preprocess_data.py > 03_preprocess_data_OUTPUT.txt
            ;;
        4)
            python 04_lda_topic_modeling.py > 04_lda_topic_modeling_OUTPUT.txt
            ;;
        5)
            python 05_train_final_lda.py > 05_train_final_lda_OUTPUT.txt
            ;;
        6)
            python 06_assign_topics.py > 06_assign_topics_OUTPUT.txt
            ;;
        7)
            python 07_naive_bayes_classification.py > 07_naive_bayes_classification_OUTPUT.txt
            ;;
        8)
            python 08_aspect_sentiment_analysis.py > 08_aspect_sentiment_analysis_OUTPUT.txt
            ;;
        9)
            python 09_generate_final_report.py > 09_generate_final_report_OUTPUT.txt
            ;;
        10)
            python 10_product_level_analysis.py > 10_product_level_analysis_OUTPUT.txt
            ;;
        11)
            python 11_user_behavior_analysis.py > 11_user_behavior_analysis_OUTPUT.txt
            ;;
        *)
            echo "Invalid argument. Please enter a number between 1 and 11."
            exit 1
            ;;
    esac
else
    echo "Invalid argument. Please enter a number between 1 and 11."
    exit 1
fi
