#RPS GAME

import pygame
from pygame import init
import os
from pygame.math import Vector2
from pygame import Rect

os.environ['SDL_Video_CENTERED'] = '1' # Center the window
init()

class UserInterface():

	def __init__(self):
		
		pygame.display.set_caption("My Game")
		self.iconImage = pygame.image.load("icon.png")
		pygame.display.set_icon(self.iconImage)
		self.clock = pygame.time.Clock()

		self.gameState = GameState()
		self.cellSize = Vector2(64,64)
		self.unitsTexture = pygame.image.load("D:/Colin Berger/Documents/Andere Benutzer/Aloïs/MA_Aloïs/Neural_Network_Github/NNFS/Optimizing_RPS/Iteration1_wihtout_ augmented_Data/fstIteration.PNG")
		self.groundTexture = pygame.image.load("icon.png")

		windowSize = self.gameState.worldSize.elementwise() * self.cellSize
		self.window = pygame.display.set_mode((int(windowSize.x),int(windowSize.y)))

		self.movePieceCommand = Vector2(0,0)
		self.running = True

	def processInput(self):

		self.moveCommand = Vector2(0,0)

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				self.running = False
				break
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_ESCAPE:
					self.running = False
					break
				elif event.key == pygame.K_d:
					self.moveCommand.x += 1
				elif event.key == pygame.K_a:
					self.moveCommand.x -= 1
				elif event.key	== pygame.K_s:
					self.moveCommand.y += 1
				elif event.key == pygame.K_w:
					self.moveCommand.y -= 1

	def update(self):

		self.gameState.update(self.moveCommand)

	def renderUnit(self,unit):

		# Location on screen
		spritePoint = unit.position.elementwise()*self.cellSize

		# Unit Texture
		texturePoint = unit.tile.elementwise()*self.cellSize
		textureRect = Rect(int(texturePoint.x), int(texturePoint.y),
						   int(self.cellWidth), int(self.cellHeight))
		self.window.blit(self.unitsTexture, spritePoint, textureRect)

	def renderGround(self, position, tile):

		# Location on the screen
		spritePoint = position.elementwise()*self.cellSize

		# Texture
		texturePoint = tile.elementwise()*self.cellSize
		textureRect = Rect(int(texturePoint.x), int(texturePoint.y),
						   int(self.cellWidth), int(self.cellWidth))
		self.window.blit(self.groundTexture, spritePoint, textureRect)

	def render(self):

		self.window.fill((0,0,0))

		for y in range(int(self.gameState.worldHeight)):
			for x in range(int(self.gameState.worldWidth)):
				self.renderGround(Vector2(x,y), self.gameState.ground[0])
		# Towers
		for unit in self.gameState.units:
			self.renderUnit(unit)


		pygame.display.update()

	def run(self):

		while self.running:
			self.processInput()
			self.update()
			self.render()
			self.clock.tick(60)

	@property
	def cellWidth(self):
		return int(self.cellSize.x)

	@property
	def cellHeight(self):
		return int(self.cellSize.y)

class GameState():

	def __init__(self):

		self.worldSize = Vector2(16,10)
		self.units = [
			Piece(self, Vector2(5,4), Vector2(1,1)),
			Tower(self, Vector2(10,3), Vector2(2,1)),
			Tower(self, Vector2(10,5), Vector2(2,3))
		]
		self.ground = [Vector2(20,31)]
		# You could doe a 2D array of each tile in the game
		# and each tile has it's untiTexture Vector

	def update(self, movePieceCommand):

		for unit in self.units:
			unit.move(movePieceCommand)

	@property
	def worldWidth(self):
		return int(self.worldSize.x)

	@property
	def worldHeight(self):
		return int(self.worldSize.y)

class Unit():

	def __init__(self, state, position, tile):
		self.state = state
		self.position = position
		self.tile = tile
		

	def move(self, moveVector):
		raise NotImplementedError()

class Piece(Unit):

	def move(self, moveVector):

		newPos = self.position + moveVector

		if  newPos.x < 0 or newPos.x >= self.state.worldWidth \
		or newPos.y < 0 or newPos.y >= self.state.worldHeight:
			return
		for unit in self.state.units:
			if newPos == unit.position:
				return

		self.position = newPos

class Tower(Unit):

	def move(self, moveVector):

		pass


	
	

unitsTexture = pygame.image.load('D:/Colin Berger/Documents/Andere Benutzer/Aloïs/MA_Aloïs/Neural_Network_Github/NNFS/Optimizing_RPS/Iteration1_wihtout_ augmented_Data/fstIteration.PNG')
window = pygame.display.set_mode((256,256))
location = pygame.math.Vector2(96,96)
rectangle = Rect(64,0,64,64)
window.blit(unitsTexture, location, rectangle)

while True:
	event = pygame.event.poll()
	if event.type == pygame.QUIT:
		break
	pygame.display.update()

userInterface = UserInterface()
userInterface.run()

pygame.quit()