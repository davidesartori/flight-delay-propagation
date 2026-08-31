from collections import defaultdict
import random
import math
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from concurrent.futures import ThreadPoolExecutor, as_completed

spark = SparkSession.builder \
    .master("local[*]") \
    .config("spark.driver.memory", "16g") \
    .appName("PySparkShell") \
    .getOrCreate()

sc = SparkContext.getOrCreate()

dir = "dataset/"
file = dir + "flights.csv"
data = spark.read.format("csv")\
                 .option("header", "true")\
                 .option("inferSchema", "true")\
                 .load(file)

sample = data.sample(fraction=0.01, seed=42)
sample.count()

delay_threshold = 15

route_delay_probability = (
    data
    .withColumn("is_delayed", F.when(F.col("DEPARTURE_DELAY") > delay_threshold, 1).otherwise(0))
    .groupBy("ORIGIN_AIRPORT", "DESTINATION_AIRPORT")
    .agg(
        F.count("*").alias("n_flights"),
        F.sum("is_delayed").alias("n_delayed")
    )
    .withColumn("delay_ratio", F.col("n_delayed") / F.col("n_flights"))
)

edges = route_delay_probability.select(
    "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "delay_ratio"
).collect()

graph = defaultdict(list)

for row in edges:
    origin = row["ORIGIN_AIRPORT"]
    dest = row["DESTINATION_AIRPORT"]
    prob = row["delay_ratio"]
    graph[origin].append((dest, prob))

graph = {
    origin: sorted(neighbors, key=lambda x: x[1], reverse=True)
    for origin, neighbors in graph.items()
}

graph_broadcast = spark.sparkContext.broadcast(graph)

def infect(current_node, graph_bc, t):
    (sample_id, u), status = current_node
    graph = graph_bc.value
    out_neighbors = graph.get(u, [])[:t] # max t neighbors   

    results = []

    for v, infecton_probability in out_neighbors:
        if random.random() < infecton_probability:
            results.append(((sample_id, v), "new"))

    results.append(((sample_id, u), "old"))
    return results


def merge_labels(label1, label2):
    if label1 == "old" or label2 == "old":
        return "old"
    return "new"

def sample_oracle(graph_bc, S, L, t, max_epochs):
    # Initialization
    infected_nodes_list = [
        ((sample_id, node), "new")
        for sample_id in range(L)
        for node in S
    ]

    infected_nodes_rdd = sc.parallelize(infected_nodes_list)

    incomplete_samples = set(range(L))
    r_t = {}
    epoch = 1

    while(len(incomplete_samples) > 0 and epoch < max_epochs):
        active_samples_infected_nodes_rdd = infected_nodes_rdd.filter(lambda node: node[0][0] in incomplete_samples)
        new_nodes = active_samples_infected_nodes_rdd.filter(lambda node: node[1] == 'new')
        
        t_d = new_nodes.flatMap(lambda row: infect(row, graph_bc, 1000))

        old_nodes = active_samples_infected_nodes_rdd.filter(lambda node: node[1] == "old")
        r_d = t_d.union(old_nodes)
        r_d = r_d.reduceByKey(merge_labels)
        
        infected_nodes_rdd = r_d

        completion_check = (
            r_d
            .map(lambda row: (
                row[0][0],
                (1, row[1] == "old")))
            .reduceByKey(lambda a, b: (
                a[0] + b[0],
                a[1] and b[1]))
            .collect()
        )
        
        for sample_id, (size, all_old) in completion_check:
            if size >= t or all_old:
                incomplete_samples.discard(sample_id)
                r_t[sample_id] = size
        

        epoch += 1
                
    n_over_threshold = sum(1 for size in r_t.values() if size >= t)
    return n_over_threshold / L


def verify_guess(graph_bc, S, n, tau, epsilon, max_epochs):
    t = tau
    total = 0.0

    while t <= n:
        L = max(10, math.ceil(8 * t * (math.log(n) ** 3) / (tau ** 2)))
 
        pi_t_L = sample_oracle(graph_bc, S, L, int(round(t)), max_epochs)
 
        total += (epsilon / (1 + epsilon)) * t * pi_t_L
 
        if total >= (1 - 2 * epsilon) * tau:
            return 1
 
        t = t * (1 + epsilon)
 
    return 0


def inf_est(graph_bc, S, n, epsilon, max_epochs):
    tau = n

    while tau > 60: #DA TOGLIERE
        tau = tau / (1 + epsilon)
 
    while tau >= len(S):
        if verify_guess(graph_bc, S, n, tau, epsilon, max_epochs) == 1:
            return tau
        tau = tau / (1 + epsilon)
 
    return 1 # we don’t reach this point

sc.setLocalProperty("spark.scheduler.mode", "FAIR")
max_workers = 10

epsilon = 0.1
n = len(graph_broadcast.value)
max_epochs = 3

airports = list(graph_broadcast.value.keys())
influence_scores = {}

already_computed = [
    'ORD', 'ATL', 'EWR', 'PHL', 'LAS', 'MCI', 'MDW', 'PBI', 'TPA', 'SMF',
    'SNA', 'CLE', 'SFO', 'JFK', 'SJC', 'DSM', 'LBB', 'AUS', 'FSD', 'MCO',
    'CAE', 'DFW', 'BQN', 'ICT', 'DTW', 'SPI', 'BMI', 'FLL', 'PSP', 'LAX',
    'BWI', 'SLC', 'MSP', 'BOS', 'PVD', 'COS', 'FAT', 'LIH', 'CVG', 'CPR',
    'CHS', 'MLI', 'DEN', 'MIA', 'FAR', 'IND', 'CAK', 'CLT', 'STL', 'BUF',
    'PSC', 'HPN', 'IAD', 'MKE', 'BNA', 'LGA', 'PHX', 'SBA', 'BPT', 'SJU',
    'SEA', 'MTJ', 'IAH', 'KOA', 'SAT', 'SWF', 'HNL', 'BHM', 'ELP', 'SBP',
    'OKC', 'SCE', 'DAL', 'GEG', 'MDT', 'MGM', 'WRG', 'PIT', 'TTN', 'MHT',
    'APN', 'TVC', 'BTV', 'MEM', 'ACY', 'DCA', 'LCH', 'ADQ', 'EUG', 'BDL',
    'CMH', 'MSY', 'VPS', 'CRP', 'RSW', 'XNA', 'GNV', 'MSO', 'AMA', 'CHA',
    'RHI',
    'JAX', 'SYR', 'PIA', 'ELM', 'OAK', 'HSV', 'LGB', 'STX', 'ANC', 'DAY',
    'ONT', 'OGG', 'LRD', 'CLL', 'HRL', 'MOB', 'BIS', 'FSM', 'ORF', 'TUL',
    'GJT', 'TLH', 'LSE', 'PDX', 'FAY', 'SGU', 'SCC', 'GRR', 'AVL', 'BZN',
    'RDU', 'TYR', 'STT', 'RIC', 'RKS', 'ABI', 'MRY', 'HOU', 'JAC', 'STC',
    'BTR', 'RNO', 'TUS', 'MFE', 'ABY', 'ASE', 'HDN', 'HYS', 'RAP', 'MOT',
    'EYW', 'MAF', 'GSO', 'OMA', 'SIT', 'ITO', 'LFT', 'TYS', 'PNS',
    'MYR', 'ECP', 'GRB', 'LAW', 'VLD', 'BRO', 'SAN', 'JAN', 'AGS', 'CWA',
    'SGF', 'BUR', 'LIT', 'FAI',
    'GPT', 'BIL', 'CMI', 'IDA', 'JLN', 'PBG', 'FWA', 'INL', 'RDD',
    'PLN', 'BQK', 'AVP', 'LWS', 'MHK', 'JNU', 'MQT', 'PIH', 'GFK',
    'SMX', 'BET', 'BGM', 'PIB', 'ROW', 'CSG', 'PPG', 'ILM', 'MMH',
    'CEC', 'MLB', 'EWN', 'OAJ', 'LAR', 'SPS', 'HOB', 'PAH', 'SUX',
    'DHN', 'ACK', 'TWF', 'MVY', 'GCK', 'BGR', 'ITH', '11298', '10721',
    '14679',
    'PWM', 'LBE', 'ISN', 'ALB', 'MKG', 'LEX',
    'ABQ', 'ILG', 'ALO', 'IAG', 'AZO', 'GSP', 'CDC', 'SDF', 'IMT',
    'BLI', 'ORH', 'SJT', 'KTN', 'ROC', 'SBN', 'CIU', 'GGG', 'EGE',
    'MSN', 'PHF', 'BFL', 'AEX', 'CNY', 'GTF', 'FNT', 'CDV', 'LAN',
    'ATW', 'MEI', 'HLN', 'MLU', 'YUM', 'MFR', 'BJI', 'DLH', 'RDM', 'SUN',
    'FCA', 'ISP', 'ERI', 'GTR', 'BOI', 'DRO', 'TXK', 'HIB', 'UST', 'PUB',
    'EVV', 'GUC', 'SHV', 'SRQ', 'CRW', 'DAB', '13303', '14100', '10140',
    '12451', '13342', '14893', '11618', '13931', '14108', '14696', '11193',
    '13796', '10154', '12339', '13244', '14771', '15016', '10868', '14869',
    '12478', '10599', '13871', '12982', '10994', '10821', '11995', '11076',
    '12892', '11267', '12992', '11697', '12323', '15024', '15412', '10423',
    '12264', '13029', '10693', '14685', '15380', '11066', '13851', '12266',
    '14107', '12954', '13830', '13495', '11481', '14576', '12758', '10469',
    '11049', '14683', '10372', '15370', '11638', '11278', '10980', '15027',
    '14635', '11996', '14307', '13422', '14908', '14524', '15096', '12953',
    '12898', '12197',
    'CHO', 'GRK', 'SAF', 'SAV', 'GCC', 'GUM',
    'RST', 'GRI', 'BRW', 'BRD', 'LNK', 'MBS', 'PSG', 'ADK',
    'ABE', 'EAU', 'CMX', 'ACV', 'DIK', 'PSE', 'ABR'
]




airports = [airport for airport in airports if airport not in already_computed]

airports = airports[100:]

def run_single(airport):
    S = [airport]
    score = inf_est(graph_broadcast, S, n, epsilon, max_epochs=max_epochs)
    return airport, score


with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(run_single, airport): airport for airport in airports}
 
    for future in as_completed(futures):
        airport = futures[future]
        try:
            airport_result, score = future.result()
            influence_scores[airport_result] = score
            print(f"{airport_result}: {score:.3f}  ({len(influence_scores)}/{n})")
        except Exception as e:
            print(f"ERRORE su {airport}: {e}")
