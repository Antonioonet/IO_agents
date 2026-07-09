
ALLOWED_TWITTER_ACTIONS = [
    ActionType.DO_NOTHING,
    ActionType.CREATE_POST,
    ActionType.CREATE_COMMENT,
    ActionType.REPOST
]

from oasis import ActionType, LLMAction

def get_available_actions() -> list: return ALLOWED_TWITTER_ACTIONS

def generate_actios(args,env) -> list:


    if args.action_mode == "natural":
        actions = {
            agent: LLMAction()
            for _, agent in env.agent_graph.get_agents()
        }
        return  actions 
    
    if args.action_mode == "prompt_probabilities":
        return [ActionType.DO_NOTHING]
    
    if args.action_mode == "autonomous":
        return [ActionType.DO_NOTHING]
    
    if args.action_mode == "calibrated":
        return [ActionType.DO_NOTHING]
    
    raise ValueError(f"Invalid action mode: {args.action_mode}")


## I suold