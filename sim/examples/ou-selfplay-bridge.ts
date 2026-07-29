/**
 * OU self-play JSONL bridge.
 *
 * This process keeps a lightweight command loop on stdin/stdout so external
 * programs (for example, Python training code) can request self-play episodes.
 *
 * @license MIT
 */

import * as readline from 'readline';
import {BattleStream, getPlayerStreams, Teams} from '..';
import type {ChoiceRequest} from '../side';
import {RandomPlayerAI} from '../tools/random-player-ai';

type RequestSeed = [number, number, number, number];
type ManagedPlayer = 'p1' | 'p2';

const MANAGED_PLAYERS: ManagedPlayer[] = ['p1', 'p2'];

interface BaseCommand {
	type: string;
	requestId?: string | number;
}

interface CreateBattleCommand extends BaseCommand {
	type: 'create_battle';
	formatid?: string;
	p1team?: string;
	p2team?: string;
	seed?: RequestSeed;
	includeLog?: boolean;
}

interface GetBattleStateCommand extends BaseCommand {
	type: 'get_battle_state';
	battleId: string;
}

interface ChooseActionCommand extends BaseCommand {
	type: 'choose_action';
	battleId: string;
	player: ManagedPlayer;
	action: string;
}

interface CloseBattleCommand extends BaseCommand {
	type: 'close_battle';
	battleId: string;
}

interface RunEpisodeCommand extends BaseCommand {
	type: 'run_episode';
	formatid?: string;
	p1team?: string;
	p2team?: string;
	seed?: RequestSeed;
	includeLog?: boolean;
}

interface RunBatchCommand extends BaseCommand {
	type: 'run_batch';
	count: number;
	formatid?: string;
	p1team?: string;
	p2team?: string;
	includeLog?: boolean;
}

interface PingCommand extends BaseCommand {
	type: 'ping';
}

type BridgeCommand =
	| CreateBattleCommand
	| GetBattleStateCommand
	| ChooseActionCommand
	| CloseBattleCommand
	| RunEpisodeCommand
	| RunBatchCommand
	| PingCommand;

interface EpisodeResult {
	formatid: string;
	winner: 'p1' | 'p2' | 'tie';
	winnerName: string;
	turns: number;
	seed?: RequestSeed;
	log?: string[];
}

interface BattleState {
	battleId: string;
	formatid: string;
	turns: number;
	done: boolean;
	winner: EpisodeResult['winner'] | null;
	winnerName: string;
	seed?: RequestSeed;
	waitingFor: ManagedPlayer[];
	requests: Partial<Record<ManagedPlayer, ChoiceRequest>>;
	errors: Partial<Record<ManagedPlayer, string[]>>;
	log?: string[];
}

const DEFAULT_FORMATID = 'gen9ou';

const DEFAULT_TEAM = packTeamFromExport(`
Great Tusk @ Leftovers
Ability: Protosynthesis
EVs: 252 HP / 4 Atk / 252 Def
Impish Nature
- Earthquake
- Rapid Spin
- Knock Off
- Stealth Rock

Gholdengo @ Air Balloon
Ability: Good as Gold
EVs: 4 Def / 252 SpA / 252 Spe
Timid Nature
IVs: 0 Atk
- Make It Rain
- Shadow Ball
- Recover
- Nasty Plot

Kingambit @ Black Glasses
Ability: Supreme Overlord
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
- Kowtow Cleave
- Sucker Punch
- Iron Head
- Swords Dance

Dragapult @ Choice Specs
Ability: Infiltrator
EVs: 4 Def / 252 SpA / 252 Spe
Timid Nature
IVs: 0 Atk
- Draco Meteor
- Shadow Ball
- Flamethrower
- U-turn

Rillaboom @ Choice Band
Ability: Grassy Surge
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
- Grassy Glide
- Wood Hammer
- Knock Off
- U-turn

Ogerpon-Wellspring (F) @ Wellspring Mask
Ability: Water Absorb
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Ivy Cudgel
- Horn Leech
- Play Rough
- U-turn
`);

function packTeamFromExport(team: string): string {
	const imported = Teams.import(team);
	if (!imported?.length) throw new Error('Failed to parse default OU team export');
	return Teams.pack(imported);
}

function normalizeTeamInput(team: string | undefined, fallback: string): string {
	if (!team) return fallback;
	const imported = Teams.import(team);
	if (!imported?.length) {
		throw new Error('Failed to parse team input. Provide a packed team, JSON team, or Pokemon Showdown export.');
	}
	return Teams.pack(imported);
}

function writeResponse(response: AnyObject) {
	process.stdout.write(`${JSON.stringify(response)}\n`);
}

function isSeedTuple(seed: unknown): seed is RequestSeed {
	if (!Array.isArray(seed) || seed.length !== 4) return false;
	return seed.every(num => Number.isInteger(num) && num >= 0);
}

function cloneRequest(request: ChoiceRequest): ChoiceRequest {
	return JSON.parse(JSON.stringify(request)) as ChoiceRequest;
}

function copyErrorMap(errors: Partial<Record<ManagedPlayer, string[]>>) {
	const copied: Partial<Record<ManagedPlayer, string[]>> = {};
	for (const player of MANAGED_PLAYERS) {
		const messages = errors[player];
		if (messages?.length) copied[player] = messages.slice();
	}
	return copied;
}

function parseBattleWinner(winnerName: string, sawTie: boolean, p1name: string, p2name: string) {
	if (sawTie) return 'tie';
	if (winnerName === p1name) return 'p1';
	if (winnerName === p2name) return 'p2';
	return null;
}

class BattleSession {
	readonly id: string;
	readonly formatid: string;
	readonly includeLog: boolean;
	readonly p1name = 'Bot 1';
	readonly p2name = 'Bot 2';
	readonly seed?: RequestSeed;
	readonly streams = getPlayerStreams(new BattleStream());

	private readonly requests: Partial<Record<ManagedPlayer, ChoiceRequest>> = {};
	private readonly errorDelta: Partial<Record<ManagedPlayer, string[]>> = {};
	private readonly logDelta: string[] = [];
	private winnerName = '';
	private sawTie = false;
	private turns = 0;
	private ended = false;
	private closed = false;
	private eventVersion = 0;
	private nextEventResolvers: Array<() => void> = [];

	constructor(id: string, command: Omit<CreateBattleCommand, 'type' | 'requestId'>) {
		this.id = id;
		this.formatid = command.formatid || DEFAULT_FORMATID;
		this.includeLog = !!command.includeLog;
		if (isSeedTuple(command.seed)) this.seed = command.seed;

		void this.consumeOmniscient();
		for (const player of MANAGED_PLAYERS) {
			void this.consumePlayer(player);
		}

		const startSpec: {formatid: string; seed?: RequestSeed} = {formatid: this.formatid};
		if (this.seed) startSpec.seed = this.seed;

		const p1team = normalizeTeamInput(command.p1team, DEFAULT_TEAM);
		const p2team = normalizeTeamInput(command.p2team, DEFAULT_TEAM);
		void this.streams.omniscient.write(`>start ${JSON.stringify(startSpec)}
>player p1 ${JSON.stringify({name: this.p1name, team: p1team})}
>player p2 ${JSON.stringify({name: this.p2name, team: p2team})}`);
	}

	private notify() {
		this.eventVersion++;
		const resolvers = this.nextEventResolvers;
		this.nextEventResolvers = [];
		for (const resolve of resolvers) resolve();
	}

	private waitForEvent() {
		return new Promise<void>(resolve => {
			this.nextEventResolvers.push(resolve);
		});
	}

	private async settle() {
		while (true) {
			const version = this.eventVersion;
			await new Promise<void>(resolve => setImmediate(resolve));
			if (version === this.eventVersion) return;
		}
	}

	private hasAnyRequest() {
		return MANAGED_PLAYERS.some(player => !!this.requests[player]);
	}

	private async consumeOmniscient() {
		for await (const chunk of this.streams.omniscient) {
			if (this.includeLog) this.logDelta.push(chunk);

			const winMatch = /\|win\|([^\n|]+)/.exec(chunk);
			if (winMatch) this.winnerName = winMatch[1].trim();
			if (/\|tie\|/.test(chunk)) this.sawTie = true;

			for (const match of chunk.matchAll(/\|turn\|(\d+)/g)) {
				const turn = Number.parseInt(match[1], 10);
				if (!Number.isNaN(turn) && turn > this.turns) this.turns = turn;
			}
			this.notify();
		}
		this.ended = true;
		this.notify();
	}

	private async consumePlayer(player: ManagedPlayer) {
		for await (const chunk of this.streams[player]) {
			for (const line of chunk.split('\n')) {
				if (!line.startsWith('|')) continue;
				const separatorIndex = line.indexOf('|', 1);
				if (separatorIndex < 0) continue;
				const command = line.slice(1, separatorIndex);
				const data = line.slice(separatorIndex + 1);
				if (command === 'request') {
					const request = JSON.parse(data) as ChoiceRequest;
					this.requests[player] = request;
					this.notify();
					continue;
				}
				if (command === 'error') {
					if (!this.errorDelta[player]) this.errorDelta[player] = [];
					this.errorDelta[player]!.push(data);
					this.notify();
				}
			}
		}
		this.notify();
	}

	private waitingForPlayers() {
		return MANAGED_PLAYERS.filter(player => {
			const request = this.requests[player];
			return !!request && !request.wait;
		});
	}

	private consumeStateDelta(): BattleState {
		const state: BattleState = {
			battleId: this.id,
			formatid: this.formatid,
			turns: this.turns,
			done: this.ended,
			winner: this.ended ? parseBattleWinner(this.winnerName, this.sawTie, this.p1name, this.p2name) : null,
			winnerName: this.winnerName,
			waitingFor: this.waitingForPlayers(),
			requests: {},
			errors: copyErrorMap(this.errorDelta),
		};
		if (this.seed) state.seed = this.seed;
		if (this.includeLog) state.log = this.logDelta.splice(0);
		for (const player of MANAGED_PLAYERS) {
			const request = this.requests[player];
			if (request) state.requests[player] = cloneRequest(request);
			delete this.errorDelta[player];
		}
		return state;
	}

	async initialize() {
		while (!this.ended && !this.hasAnyRequest()) {
			await this.waitForEvent();
		}
		await this.settle();
		return this.consumeStateDelta();
	}

	async getState() {
		await this.settle();
		return this.consumeStateDelta();
	}

	async chooseAction(player: ManagedPlayer, action: string) {
		if (this.closed) throw new Error(`Battle ${this.id} is already closed`);
		delete this.requests[player];
		void this.streams[player].write(action.trim());
		await this.settle();
		return this.consumeStateDelta();
	}

	async close() {
		if (this.closed) return;
		this.closed = true;
		await this.streams.omniscient.writeEnd();
		await this.settle();
	}
}

const activeBattles = new Map<string, BattleSession>();
let nextBattleId = 1;

function createBattleId() {
	return `battle-${nextBattleId++}`;
}

function getBattleSession(battleId: string) {
	const session = activeBattles.get(battleId);
	if (!session) throw new Error(`Unknown battleId: ${battleId}`);
	return session;
}

async function runEpisode(command: Omit<RunEpisodeCommand, 'type'>): Promise<EpisodeResult> {
	const formatid = command.formatid || DEFAULT_FORMATID;
	const p1name = 'Bot 1';
	const p2name = 'Bot 2';
	const p1team = normalizeTeamInput(command.p1team, DEFAULT_TEAM);
	const p2team = normalizeTeamInput(command.p2team, DEFAULT_TEAM);

	const streams = getPlayerStreams(new BattleStream());
	const p1 = new RandomPlayerAI(streams.p1);
	const p2 = new RandomPlayerAI(streams.p2);

	void p1.start();
	void p2.start();

	const startSpec: {formatid: string; seed?: RequestSeed} = {formatid};
	if (isSeedTuple(command.seed)) startSpec.seed = command.seed;

	void streams.omniscient.write(`>start ${JSON.stringify(startSpec)}
>player p1 ${JSON.stringify({name: p1name, team: p1team})}
>player p2 ${JSON.stringify({name: p2name, team: p2team})}`);

	const log: string[] = [];
	let winnerName = '';
	let sawTie = false;
	let turns = 0;

	for await (const chunk of streams.omniscient) {
		if (command.includeLog) log.push(chunk);

		const winMatch = /\|win\|([^\n|]+)/.exec(chunk);
		if (winMatch) winnerName = winMatch[1].trim();
		if (/\|tie\|/.test(chunk)) sawTie = true;

		for (const match of chunk.matchAll(/\|turn\|(\d+)/g)) {
			const turn = Number.parseInt(match[1], 10);
			if (!Number.isNaN(turn) && turn > turns) turns = turn;
		}
	}

	let winner: EpisodeResult['winner'] = 'tie';
	if (sawTie) winner = 'tie';
	if (winnerName === p1name) winner = 'p1';
	if (winnerName === p2name) winner = 'p2';

	const result: EpisodeResult = {
		formatid,
		winner,
		winnerName,
		turns,
	};
	if (isSeedTuple(command.seed)) result.seed = command.seed;
	if (command.includeLog) result.log = log;
	return result;
}

function parseCommand(raw: string): BridgeCommand {
	let parsed: unknown;
	try {
		parsed = JSON.parse(raw);
	} catch {
		throw new Error('Invalid JSON input');
	}
	if (!parsed || typeof parsed !== 'object') throw new Error('Command must be a JSON object');
	const command = parsed as Partial<BridgeCommand>;
	if (!command.type || typeof command.type !== 'string') throw new Error('Missing command type');
	if (!['ping', 'create_battle', 'get_battle_state', 'choose_action', 'close_battle', 'run_episode', 'run_batch'].includes(command.type)) {
		throw new Error(`Unknown command type: ${command.type}`);
	}
	return command as BridgeCommand;
}

async function handleCommand(command: BridgeCommand) {
	if (command.type === 'ping') {
		writeResponse({type: 'pong', requestId: command.requestId});
		return;
	}

	if (command.type === 'create_battle') {
		const battleId = createBattleId();
		const session = new BattleSession(battleId, command);
		activeBattles.set(battleId, session);
		const state = await session.initialize();
		writeResponse({type: 'battle_created', requestId: command.requestId, battleId, state});
		return;
	}

	if (command.type === 'get_battle_state') {
		const session = getBattleSession(command.battleId);
		const state = await session.getState();
		writeResponse({type: 'battle_state', requestId: command.requestId, battleId: command.battleId, state});
		return;
	}

	if (command.type === 'choose_action') {
		if (!command.action || typeof command.action !== 'string') {
			throw new Error('choose_action requires a non-empty action string');
		}
		if (!MANAGED_PLAYERS.includes(command.player)) {
			throw new Error(`Unsupported player id: ${String(command.player)}`);
		}
		const session = getBattleSession(command.battleId);
		const state = await session.chooseAction(command.player, command.action);
		writeResponse({type: 'action_result', requestId: command.requestId, battleId: command.battleId, state});
		return;
	}

	if (command.type === 'close_battle') {
		const session = getBattleSession(command.battleId);
		await session.close();
		activeBattles.delete(command.battleId);
		writeResponse({type: 'battle_closed', requestId: command.requestId, battleId: command.battleId});
		return;
	}

	if (command.type === 'run_episode') {
		const result = await runEpisode(command);
		writeResponse({type: 'episode_result', requestId: command.requestId, result});
		return;
	}

	if (!Number.isInteger(command.count) || command.count <= 0) {
		throw new Error('run_batch requires count as a positive integer');
	}

	const episodes: EpisodeResult[] = [];
	for (let i = 0; i < command.count; i++) {
		episodes.push(await runEpisode(command));
	}
	writeResponse({type: 'batch_result', requestId: command.requestId, episodes});
}

async function main() {
	writeResponse({
		type: 'ready',
		defaultFormatid: DEFAULT_FORMATID,
		supports: [
			'ping',
			'create_battle',
			'get_battle_state',
			'choose_action',
			'close_battle',
			'run_episode',
			'run_batch',
		],
	});

	const rl = readline.createInterface({
		input: process.stdin,
		crlfDelay: Infinity,
	});

	for await (const line of rl) {
		const trimmed = line.trim();
		if (!trimmed) continue;
		try {
			const command = parseCommand(trimmed);
			await handleCommand(command);
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error);
			writeResponse({type: 'error', message});
		}
	}
}

void main();