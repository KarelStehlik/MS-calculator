def help():
    return '''
crit [chance in %], [multiplier in %], [joker=False]:
    returns the average damage multiplier from crits
titan [attack]:
    returns new attack after obtaining titan
unTitan [attack]:
    returns what attack would be without titan
accel [cdr in %], [old movespeed in %], [new movespeed in %]:
    returns new cdr in % after increasing move speed with accelerator
gaia [hp]:
    returns attack value gained from gaia
equation [string]:
    attempts to solve an equation with variable x.
    if an expression is entred containing " = ", it is considered an equation.
hydraEclipse [hitcount, hasHydra, hasEclipse, hasRose, hasSword]
    returns damage multiplier from items which interact with current health
    enemies which you would otherwise kill with a given # of hits
(deprecated) artifactForce [# of desired artifacts]
    returns the chance that you will get all desired legendaries in 1 run
    no longer valid as legendary artifacts can repeat
plot [expr, minX, maxX]
multiplot [minX, maxX, *exprs]
plot3d [expr, minX, maxX, minY, maxY]
'''

def chatgptHydra(n):
    '''
    Written by chatGpt using the following prompt:
    >    imagine a game where you hit an enemy to kill it.
    >    Every hit deals the same amount of damage.
    >    You know only how many hits it takes to kill an enemy.
    >    Based on this number, write a python function that
    >    calculates the multiplicative increase in your average
    >    hit damage when you obtain the following effect:
    >    "Hits have added damage equal to 0.5% of the target's
    >    current health. The total damage added by this cannot
    >    exceed 30% of the target's maximum health"
    It's horribly wrong, but I keep it around beause it's kinda interesting.
    '''
    D = 1;  '''perfect''' # Assume initial damage per hit is 1 for simplicity
    total_health = D * n; '''good'''
    
    # Calculate total damage without the effect
    total_damage_without_effect = total_health; '''yes'''
    
    # Calculate total damage with the effect
    total_damage_with_effect = 0
    current_health = total_health; '''mhm'''
    
    for i in range(n):
        added_damage = 0.005 * current_health; '''correct'''
        # Ensure added damage doesn't exceed 30% of max health
        added_damage = min(added_damage, 0.3 * total_health);
        '''I get the problem here, i was a bit ambiguous.
            Still, I feel like you should be able to deduce that 0.5%
            current health probably isn't going to be more than 30% max,
            so I probably meant the total damage from all hits?
        '''
        
        total_damage_with_effect += D + added_damage; '''alright'''
        current_health -= D; '''now, where did the extra damage go?'''
    
    # Calculate the average damage per hit
    average_damage_with_effect = total_damage_with_effect / n; '''all good from here'''
    
    # Since D is 1, the multiplicative increase is the same as the average damage
    multiplicative_increase = average_damage_with_effect / D
    
    return multiplicative_increase


def hydraEclipse(baseHits, hasHydra, hasEclipse, hasRose=False, hasSword=False):
    # This assumes that hydra damage is increased by eclipse and rose,
    # and that the final damage after increases counts towards hydra's cap
    # these assumptions can be modified with the 2 commented lines.
    hp = baseHits
    actualHits = 0
    totalHydraDamage = 0
    hydraCap = baseHits*0.3
    
    while hp> (baseHits*0.2 if hasSword else 0):
            spell = 1
            hydra = min(hydraCap-totalHydraDamage, hp*0.005) if hasHydra else 0
            eclipse = 1 + (min(0.45, (baseHits-hp)/baseHits) if hasEclipse else 0)
            rose = 1.2 if hp/baseHits>0.9 and hasRose else 1
            hp -= (spell+hydra)*eclipse*rose # hydra is increased by eclipse and rose
            totalHydraDamage += hydra*eclipse*rose # final damage after increases counts towards hydra's cap
            actualHits+=1
    damageDealt = baseHits-hp + (0.2*baseHits if hasSword else 0)
    return damageDealt/actualHits


def crit(chance, mult, joker=False):
    chance = min(1, chance/100)
    mult /= 100
    if not joker:
        return 1 + chance *(mult-1)
    return 1 + chance * (mult-1 + chance/2*(mult-1))

def titan(attack):
    return attack + (attack-100)*.5

def unTitan(attack):
    return 30 + (attack-30)/1.5

def accel(cdr, oldspeed, newspeed):
    cdr/=100
    oldspeed/=100
    newspeed/=100
    oldAccel = 1-(oldspeed-1)/3
    newAccel = 1-(newspeed-1)/3
    oldcd = 1-cdr
    return 100*(1-(oldcd/oldAccel*newAccel))

def gaia(hp):
    return int((hp+100)/20)*30*.03

def gaiaUnround(hp):
    return (hp+100)/20*30*.03

def equation(eq):
    eq = f'({eq.replace(" = ", ")-(")})'
    
    def f(x):
        return eval(eq)
    
    def f_prime(x, h=1e-6):
        return (f(x + h) - f(x - h)) / (2 * h)
    
    x = 1.0  # Initial guess
    tolerance = 1e-6
    max_iterations = 1000
    
    for i in range(max_iterations):
        error = f(x)
        derivative = f_prime(x)
        
        if abs(derivative) < tolerance:  # Avoid division by a small number
            derivative = 0.1
        
        step = -error / derivative
        x += step
        
        if abs(step) < tolerance:  # Convergence criterion
            return f"Solution is x={x:.6f} with error {error:.3E} after {i+1} iterations"
    
    return f"Failed to converge after {max_iterations} iterations"


def artifactForce(x):
    chance=1
    buckets=27
    trapped=0
    for i in range(x):
        chance *= 1-trapped/buckets
        trapped+=2
        buckets-=1
    return chance

def plot(expr, minX, maxX):
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.linspace(minX, maxX, 200)
    y = [eval(expr) for x in x]

    fig, ax = plt.subplots()
    ax.plot(x, y)
    plt.show()

def multiplot(minX, maxX, *exprs):
    import matplotlib.pyplot as plt
    import numpy as np
    
    x = np.linspace(minX, maxX, 200)

    for expr in exprs:
        name,value=expr.split("|")
        y = [eval(value) for x in x]
        plt.plot(x, y, label=name)
        
    plt.legend()
    plt.show()

def plot3d(expr, minX, maxX, minY, maxY):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import cm

    plt.style.use('Solarize_Light2')

    # Make data
    X = np.arange(minX, maxX, (maxX-minX)/50)
    Y = np.arange(minY, maxY, (maxY-minY)/50)
    X, Y = np.meshgrid(X, Y)

    func = lambda x,y: eval(expr)
    func=np.vectorize(func)
    Z=func(X,Y)

    # Plot the surface
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, Z, cmap=cm.plasma)

   # ax.set(xticklabels=[], yticklabels=[], zticklabels=[])

    plt.show()

running=True
while running:
    try:
        print(">>> ", end='')
        expr=input()
        if expr.count(" = ") == 1:
            print(equation(expr))
        else:
            print(eval(expr))
    except KeyboardInterrupt as e:
        running=False
    except Exception as e:
        print("Heavens above, disaster has struck!")
        print(e.args)
