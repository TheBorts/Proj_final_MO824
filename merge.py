import numpy as np
import pandas as pd
import csv

def merge_results(base, to_merge, output_file):
    """
        For every instance in to_merge that isnt in base, add it to base.
        Finally, save the merged dataframe to output_file.

        The header from to_merge is:
        config,file,n,k,alpha,construct_mode,ls_mode,reactive_alphas,reactive_block,sample_size,iterations,time_limit_s,timed_out,max_value,size,feasible,time_s,elements

        And the header from base is:
        config,file,n,k,alpha,construct_mode,ls_mode,reactive_alphas,reactive_block,sample_size,iterations,time_limit_s,timed_out,max_value,size,feasible,time_s,time_to_solution_s,elements

        The time_to_solution_s column is missing in to_merge, so we will add it with the same values as time_s.
    """

    df_base = pd.read_csv(base)
    df_to_merge = pd.read_csv(to_merge)

    # Add time_to_solution_s column to df_to_merge
    df_to_merge['time_to_solution_s'] = df_to_merge['time_s']

    # Find instances in to_merge that are not in base
    merged_df = pd.concat([df_base, df_to_merge[~df_to_merge['file'].isin(df_base['file'])]], ignore_index=True)

    # Save the merged dataframe to output_file
    merged_df.to_csv(output_file, index=False, quoting=csv.QUOTE_NONNUMERIC)

def fix_csv_formatting(input_file, output_file):
    """
        Fix the CSV formatting by ensuring all fields are properly quoted.
        Sometimes there exists an 8th column, which breaks the CSV format.
        This function will read the CSV, and rewrite it without the 8th column.
    """
    with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile, quoting=csv.QUOTE_NONNUMERIC)

        for row in reader:
            # Only take the first 7 columns
            fixed_row = row[:7]
            writer.writerow(fixed_row)

if __name__ == "__main__":
    fix_csv_formatting("src/literature_heuristics/results/clustering_results_merge.csv", "src/literature_heuristics/results/clustering_results_fixed.csv")
