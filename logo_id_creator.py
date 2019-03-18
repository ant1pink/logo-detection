import csv
from math import sqrt
import os

MAXINT = 999999999
threshold = 100

def distance(x1, y1, x2, y2):
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    return sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))

def create_id(infile, outfile):
    with open(infile, 'rU') as csvfile:
        data_rows = csv.DictReader(csvfile, delimiter=',', 
                        fieldnames=['frame_id', 'brand', 'location', 'size', 'percentage', 'x', 'y', 'confidence', 'count'])
        
        logo_details = {}
        id_counter = 0        
        # Note: All frame_ids in csv file are in ascending order.
        for row in data_rows:
            if row['frame_id'] == 'frame_id':
                continue
        
            frame_id = int(row['frame_id'].strip())
            if frame_id not in logo_details:    # It's a new frame.
                logo_details[frame_id] = {}
                
            key = '{}+{}'.format(row['brand'], row['location'])
            
            if (frame_id - 1) in logo_details and key in logo_details[frame_id-1]:  
                # If it's not the first frame and the logo appeares in the previous frame...
                min_distance = MAXINT
                logo_id = -1
                for logo in logo_details[frame_id-1][key]:
                    temp = distance(logo['x'], logo['y'], row['x'], row['y'])
                    if temp < min_distance:
                        min_distance = temp
                        logo_id = logo['logo_id']
                if logo_id != -1 and min_distance < threshold:
                    row['logo_id'] = logo_id
                else:
                    row['logo_id'] = id_counter
                    id_counter += 1
            else:
                row['logo_id'] = id_counter
                id_counter += 1
                
            if key not in logo_details[frame_id]:
                logo_details[frame_id][key] = [row]
            else:
                logo_details[frame_id][key].append(row)
                
    with open(outfile, 'w', newline='') as fout:
        csvfile = csv.writer(fout, delimiter=',')
        csvfile.writerow(['frame_id', 'brand', 'location', 'size', 'percentage', 'x', 'y', 'confidence', 'count', 'logo_id'])
        
        frame_ids = sorted(logo_details.keys())
        for i in frame_ids:
            print("Processing frame " + str(i))
            for k, v_list in logo_details[i].items():
                for obj in v_list:
                    new_row = []
                    for col in ['frame_id', 'brand', 'location', 'size', 'percentage', 'x', 'y', 'confidence', 'count', 'logo_id']:
                        new_row.append(obj[col])
                    csvfile.writerow(new_row)

if __name__ == '__main__':
    infile = 'test1.csv'
    _, tail = os.path.split(infile)
    outfile = '%s_with_logoid.csv'%(tail.split('.')[0])
    create_id(infile, outfile)