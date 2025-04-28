Other=['Shoes','napkin','plastic bags','milk tea','nappies','chewing gums','cups','cigarette','preservative film','socks','lighters','toothpicks','masks']
Foodwaste=['Egg shells','prawns','green vegetables','watermelon rind','chocolates','hot pot base','Egg shellsdddaa','rice','meat','banana skin','breads']
Recyclables=['glasses','knife','dolls','power bank','pans','newspaper','outlet','bags','wine bottle','cans','spike']
Hazardous=['thermometers','bulbs','pill']


def main(name):
    if name in Other :
        tp='Other waste'
        content='Other waste refers to domestic waste other than recyclables, hazardous waste and food waste. i.e. waste that is mainly collected and treated by the current sanitation system.' \
                'They are generally disposed of by landfill, incineration, etc. Some of them can also be solved by using biodegradation methods, such as putting earthworms. Other waste is recyclables,' \
                'A type of waste remaining from food waste and hazardous waste.'
        return [name, tp, content]

    if name in Hazardous :
        tp = 'Hazardous waste'
        content = 'Hazardous waste means substances in domestic waste that are directly or potentially harmful to human health or the natural environment.\
                  It must be collected, transported and stored separately, and be subject to special safe treatment by professional organisations recognised by the environmental authorities.' \
                  'Common hazardous waste includes waste lamps, waste paint, pesticides, discarded cosmetics, expired medicines, waste batteries,' \
                  'Spent light bulbs, wastewater silver thermometers, etc. Hazardous waste needs to be disposed of safely in a special and correct way.'
        return [name, tp, content]
    if name in Recyclables:
        tp = 'Recyclables'
        content = 'Recyclables are renewable resources, referring to uncontaminated waste from domestic waste that is suitable for recovery and recycling.' \
                  ' Five main categories, including waste electrical and electronic products, waste paper, waste plastics, waste glass and waste metals.' \
                  'It is the main task of domestic waste separation and an important factor affecting waste reduction at this stage.'
        return [name, tp, content]
    if name in Foodwaste:
        tp = 'Food waste'
        content = 'Food waste refers to waste generated from the daily lives of residents and from activities such as food processing, catering services, and institutional catering.' \
                  'This includes discarding unused leaves, leftovers, fruit peels, egg shells, tea dregs, bones (chicken bones, fish bones), etc.' \
                  'The main sources are home kitchens, restaurants, hotels, canteens, markets and other industries related to food processing.'
        return [name, tp, content]



if __name__ == '__main__':
    res=main('')
    for i in res:
        print(i)

