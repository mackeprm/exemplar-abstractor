def toLocalId(input_id):
    return input_id.split("/")[len(input_id.split("/")) - 1]


def truncate(input_string, length):
    return (input_string[:length] + '..') if len(input_string) > length else input_string
