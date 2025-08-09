import os
import sys

def parse_named_args() -> dict:
    """
    Parses command-line arguments of the form:
        --argName value {}'argName': 'value'}
        -flagName       {'flagName': True}
        
    If --argName is specified without a following value then it will be assigned None

    Returns:
        dict: Dictionary of parsed arguments.
    """
    
    args = sys.argv[1:]  # Skip script name
    args_parsed = {}
    i = 0

    while i < len(args):
        arg = args[i]

        if arg.startswith('--'):
            key = arg[2:]
            # Check if next item exists and isn't another flag
            if i + 1 < len(args) and not args[i + 1].startswith('-'):
                args_parsed[key] = args[i + 1]
                i += 2
            else:
                args_parsed[key] = None
                i += 1

        elif arg.startswith('-'):
            key = arg[1:]
            args_parsed[key] = True
            i += 1

        else:
            # Positional or unexpected argument, skip.
            i += 1  

    return args_parsed

def read_and_parse_config_file(filename: os.PathLike) -> dict:
    """
    Parses a configuration text file into a dictionary.

        - Ignores blank lines and comments (marked with '#').
        - Splits each valid line into a key-value pair using the first ':' character.
        - Strips whitespace from keys and values.
        - Converts values to float, or int, or keeps as strings.

    Parameters:
        filename (os.PathLike): The path to the text file to be parsed.

    Returns:
        dict: A dictionary of the configuration file contents.
    """

    result = {}
    with open(filename, 'r') as file:
        
        for line in file:
            
            # Remove comments and skip blank lines
            line = line.split('#')[0].strip()
            if not line:
                continue  

            # Handle valid lines 
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()

                # Try to convert value to int or float and failing that 
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass  # Keep as string if conversion fails

                result[key] = value
                
    return result