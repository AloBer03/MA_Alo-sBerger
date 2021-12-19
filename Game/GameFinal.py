#Games RPS

import pygame
from nna import *
import cv2
import os
import numpy as np
from pygame import Vector2

# Center the window
os.environ['SDL_Video_CENTERED'] = '1'

pygame.init()
pygame.font.init()

# Load the already optimized model
model = Model.load('Network9.model')
# Labels 
RPS_labels = [
	'Rock',
	'Paper',
	'Scissors',
	'Rock'
]

class Game:

	def __init__(self):

		# Create Windwow
		pygame.display.set_caption("My Game")
		self.iconImage = pygame.image.load("icon.png")
		pygame.display.set_icon(self.iconImage)
		# Set clock
		self.clock = pygame.time.Clock()
		self.running = True
		# Setting up The window
		self.size = Vector2(800,600)
		self.window = pygame.display.set_mode(Vector2(800,600)) # it might be only (800,600) instead of Vector2(800,600))
		# Set Fonts
		self.hugeFont = pygame.font.SysFont(pygame.font.get_default_font(), 84)
		self.bigFont = pygame.font.SysFont(pygame.font.get_default_font(), 72)
		self.middleFont = pygame.font.SysFont(pygame.font.get_default_font(), 48)
		self.smallFont = pygame.font.SysFont(pygame.font.get_default_font(), 36)
		self.tinyFont = pygame.font.SysFont(pygame.font.get_default_font(), 24)
		self.noteFont = pygame.font.SysFont(pygame.font.get_default_font(), 14)
		# Set variables to default
		self.scene = 1 # scene
		self.sc = False
		self.mbut = (False, False, False) # Mouse Pressed
		self.object = 4
		cv2.imwrite('cam/imPrev.png', cv2.imread("icon.png"))
		
		# Which model to use (if not My then use keras one)
		self.model_use = "My"
		# Create Buttons
		self.buttons = [Button(self, (self.width/2,250), text="Play", mode="CENTER", font=self.middleFont, scen=1),
						Button(self, (self.width/2,300), text="Help", mode="CENTER", font=self.smallFont, scen=1),
						Button(self, (40,self.height-20), text="<-Back", mode="CENTER", font=self.tinyFont, scen=[3,4])

		]
	# Input processing
	def processInput(self):

		# Get all the inputs
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				self.running = False
				break
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_ESCAPE:
					self.running = False
					break

		# Mouse
		self.last_mbut = self.mbut
		self.mpos = pygame.mouse.get_pos() # For button (if on it => on it = on)
		self.mbut = pygame.mouse.get_pressed()

		# If on Button
		for button in self.buttons:
			if button.on_it():
				button.on = True
			else:
				button.on = False

		# If needed take Images form camera
		if self.scene == 3:
			# cv2.namedWindow("Preview")
			vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)

			# Try to get the first frame
			if vc.isOpened(): 
				rval, self.frame = vc.read()
			else:
				rval = False

			# Set frame to current frame of the camera
			rval, self.frame = vc.read()
			#cv2.imshow("Preview", self.frame)
			
			# Resize
			self.frame = cv2.resize(self.frame, (28,28))
			# Convert into gray
			self.grayframe = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
			# cv2.imwrite('cam/im.png', self.frame)

		if self.scene == 4:
			#cv2.namedWindow("Preview")
			vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)
			
			# Try to get the first frame
			if vc.isOpened(): 
				rval, self.frame = vc.read()
			else:
				rval = False

			# Set frame to current frame of the camera
			rval, self.frame = vc.read()

			# cv2.imshow("Preview", self.frame)
			# Resize
			fram = cv2.resize(self.frame, (300, 300))
			# self.grayframe = cv2.cvtColor(fram, cv2.COLOR_BGR2GRAY)
			cv2.imwrite('cam/imPrev.png', fram)
			
			
	# Update the game
	def update(self):

		if self.scene == 3:
			image_datas = []
			#data = cv2.imread('cam/im.png', cv2.IMREAD_GRAYSCALE)
			data = self.grayframe

			# Pass the Image through the network
			if self.model_use == "My":
				image_data = (data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5
				self.confidences = model.predict(image_data)
				predictions = model.output_layer_activation.predictions(self.confidences)
				self.object_class = predictions[0]
				self.object = RPS_labels[predictions[0]]
			else:
				image_data = np.expand_dims(data, -1)
				image_data = np.expand_dims(image_data,0)
				image_data = image_data.astype('float16')
				image_datas.append(image_data)
				image_datas.append(image_data)
				image_datas = np.array(image_datas)
				self.confidences = keras_model.predict(image_data)
				self.object_class = np.argmax(self.confidences)
				self.object = RPS_labels[self.object_class]

			print(self.confidences)
			print(self.object_class, self.object)

		# Update the scene
		if self.buttons[0].on and self.mbut[0] and (not self.last_mbut[0]) and self.buttons[0].scen == self.scene:
			self.scene = 2
		if self.buttons[1].on and self.mbut[0] and (not self.last_mbut[0]) and self.buttons[1].scen == self.scene:
			self.scene = 4
		for sce in self.buttons[2].scen:
			if sce == self.scene:
				self.sc = True
		if self.buttons[2].on and self.mbut[0] and (not self.last_mbut[0]) and self.sc:
			self.scene = 1
		self.sc = False

	# Create textfield 
	def text(self, surface, text, pos, font, col, mode="LEFT", antialias=True, background=None):
		if mode=="CENTER":
			textSurface = font.render(str(text), antialias, col, background)
			size = font.size(text)
			surface.blit(textSurface, (pos[0] - size[0]/2, pos[1] - size[1]/2))
			return size
		elif mode=="LEFT":
			text = font.render(str(text), antialias, col, background)
			surface.blit(text, pos)
			return size
		elif mode=="RIGHT":
			textSurface = font.render(str(text), antialias, col, background)
			size = font.size(text)
			surface.blit(textSurface, (pos[0] - size[0], pos[1] - size[1]))
			return size
		else:
			print("Not a valid Mode")

	# Render menu scene
	def render_menu(self):
		# Set Font
		sizePlay = self.smallFont.size("Play")
		# If mouse over Button make the background gray
		for button in self.buttons[:2]:
			if button.on:
				back = (200,200,200)
			else: 
				back = None
			button.render(back=back)
		# Render Titel
		self.text(self.window, "Rock, Paper, Scissors", (self.width/2,80), self.bigFont, (0,0,0), mode="CENTER" )
		
	# Render the animation for starting the Game
	def render_animationStart(self):

		self.text(self.window, "Ready?", (self.width/2,self.height/2), self.bigFont, (0,0,0), mode="CENTER" )
		pygame.display.update()
		pygame.time.delay(2000)
		self.window.fill((255,255,255))
		self.text(self.window, "Rock", (self.width/2,self.height/2), self.bigFont, (0,0,0), mode="CENTER" )
		pygame.display.update()
		pygame.time.delay(1000)
		self.window.fill((255,255,255))
		self.text(self.window, "Paper", (self.width/2,self.height/2), self.bigFont, (0,0,0), mode="CENTER" )
		pygame.display.update()
		pygame.time.delay(1000)
		self.window.fill((255,255,255))
		self.text(self.window, "Scissors", (self.width/2,self.height/2), self.bigFont, (0,0,0), mode="CENTER" )
		pygame.display.update()
		pygame.time.delay(500)
		self.scene = 3

	# Render the playing Scene
	def render_play(self):
		# Display the object which counters the recognized one
		if self.object_class != 4:
			answer = RPS_labels[self.object_class + 1]
			self.text(self.window, str(answer), (self.width/2,self.height/2), self.hugeFont, (0,0,0), mode="CENTER" )
			pygame.display.update()
		if self.buttons[2].on:
			back = (200,200,200)
		else: 
			back = None
		self.buttons[2].render(back=back)

	# Render Help scene
	def render_help(self):

		self.text(self.window, "Put the back of your hand in front of the camera",
				  (self.width/2,100), self.middleFont, (0,0,0), mode="CENTER")
		self.text(self.window, "and make sure you roughly fill the frame",
				 (self.width/2,150), self.middleFont, (0,0,0), mode="CENTER")
		self.text(self.window,"Note: It works best when the hand comes from the bottom (relative to the camera)", 
				  (self.width/2,560), self.noteFont, (0,0,0), mode="CENTER")
		img = pygame.image.load("cam/imPrev.png")
		self.window.blit(img, (250,250))
		if self.buttons[2].on:
			back = (200,200,200)
		else: 
			back = None
		self.buttons[2].render(back=back)

		
	# Main Render
	def render(self):

		# Decide which scen has to be rendered
		self.window.fill((255,255,255))
		if self.scene == 1:
			self.render_menu()
		elif self.scene == 2:
			self.render_animationStart()
		elif self.scene == 3:
			self.render_play()
		elif self.scene == 4:
			self.render_help()
		# Testing scene?
		pygame.display.update()

	def run(self):

		# Loop through the 3 main function
		while self.running:
			self.processInput()
			self.update()
			self.render()
			# set clockspeed to 60 tick per second
			self.clock.tick(60)

	@property
	def width(self):
		return int(self.size.x)

	@property
	def height(self):
		return int(self.size.y)

# Buttons
class Button():

	def __init__(self, interface, pos, width=None, height=None, text=None, mode="LEFT", font=pygame.font.SysFont(pygame.font.get_default_font(), 36), scen=None):
		self.interface = interface
		self.mode = mode
		self.pos = pos
		self.x = pos[0]
		self.y = pos[1]
		self.width = width
		self.height = height
		self.text = text
		self.font = font
		self.on = False
		self.scen = scen

	# Determine if the mouse is on the button
	def on_it(self):
		mpos = pygame.mouse.get_pos()
		sizeText = self.font.size(self.text)
		if self.mode=="CENTER":
			if self.x - sizeText[0]/2 < mpos[0] and \
			   self.x + sizeText[0]/2 > mpos[0] and \
			   self.y - sizeText[1]/2 < mpos[1] and \
			   self.y + sizeText[1]/2 > mpos[1]:
				return True
			else:
				return False
		elif mode=="LEFT":
			if self.x < self.mpos[0] and \
			   self.x + sizeText[1] > mpos[0] and \
			   self.y < self.mpos[1] and \
			   self.y + sizeText[1] > mpos[1]:
				return True
			else:
				return False
		elif mode=="RIGHT":
			if self.x - sizeText[0] < mpos[0] and \
			   self.x > self.mpos[0] and \
			   self.y - sizeText[1] < mpos[1] and \
			   self.y > self.mpos[1]:
				return True
			else:
				return False

	# render the button
	def render(self, col=(0,0,0), back=None):
		self.interface.text(self.interface.window, self.text, self.pos, self.font, col, mode=self.mode, background=back)



game = Game()
game.run()

